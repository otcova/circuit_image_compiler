use crate::utils::sync_state::*;
use crate::{circuit::*, utils::clock::Clock};
use std::{sync::Arc, thread, time::Duration};

pub struct CircuitRuntime {
    pub engine: Box<dyn CircuitEngine>,
    pub state: CircuitState,

    // If some, the simulation will be ticked in inverals of Duration.
    pub tick_interval: Option<Duration>,

    exit: bool,
}

impl Clone for CircuitRuntime {
    fn clone(&self) -> Self {
        Self {
            engine: self.engine.clone_dyn(),
            state: self.state.clone(),
            tick_interval: self.tick_interval,
            exit: self.exit,
        }
    }
    fn clone_from(&mut self, other: &Self) {
        if self.engine.name() != other.engine.name() {
            self.engine = other.engine.clone_dyn();
        }
        self.state.clone_from(&other.state);
        self.tick_interval = other.tick_interval;
        self.exit = other.exit;
    }
}

pub struct CircuitRunner {
    is_paused: bool,

    // Runtime thread will only block the mutex quickly to update the state with memcpy,
    // wich will happen at most every few ms.
    runtime: Arc<SyncState<CircuitRuntime>>,
}

impl CircuitRunner {
    /// How often the runner thread should update global state.
    /// 8ms corresponds to about twice per frame in 60fps.
    const UPDATE_INTERVAL: Duration = Duration::from_millis(8);

    pub fn new(state: CircuitState, engine: Box<dyn CircuitEngine>) -> Self {
        let shared_runtime = Arc::new(SyncState::new(CircuitRuntime {
            engine: engine.new_dyn(&state.image),
            state: state.clone(),
            tick_interval: None,
            exit: false,
        }));
        let shared_runtime_clone = shared_runtime.clone();

        thread::spawn(move || {
            let mut runtime = Local::new(CircuitRuntime {
                engine,
                state,
                tick_interval: None,
                exit: false,
            });
            let mut clock = Clock::new(Self::UPDATE_INTERVAL);

            while !runtime.exit {
                // Run ticks and sleep thread for UPDATE_INTERVAL time
                if let Some(dt) = runtime.tick_interval {
                    clock.run_ticks(dt, |n| {
                        // Do at most 256 at a time to not exceed UPDATE_INTERVAL.
                        let n = n.min(256);
                        runtime.tick_n(n as u64);
                        n
                    });
                }

                let mut tick_interval_changed = false;
                let prev_tick_interval = runtime.tick_interval;

                shared_runtime.wait_while(&mut runtime, |s| {
                    let wait = !s.exit && s.tick_interval.is_none();
                    if !tick_interval_changed && s.tick_interval != prev_tick_interval {
                        tick_interval_changed = true;
                    }
                    wait
                });

                if tick_interval_changed {
                    clock.reset();
                }
            }
        });

        Self {
            runtime: shared_runtime_clone,
            is_paused: true,
        }
    }

    pub fn tick_n(&mut self, n: u64) {
        self.runtime.overwrite(|runtime| {
            runtime.tick_interval = None;
            runtime.tick_n(n);
        });
        self.is_paused = true;
    }

    pub fn set_tick_interval(&mut self, interval: Option<Duration>) {
        self.runtime.overwrite(|r| r.tick_interval = interval);
        self.is_paused = interval.is_none();
    }

    pub fn is_paused(&self) -> bool {
        self.is_paused
    }

    pub fn set_engine(&self, engine: Box<dyn CircuitEngine>) {
        self.runtime.overwrite(|r| r.engine = engine);
    }

    /// f should return quickly since it acquires a lock.
    pub fn get<R>(&self, f: impl FnOnce(&CircuitRuntime) -> R) -> R {
        self.runtime.get(|r| f(r))
    }

    /// A lock is acquired while f runs.
    /// Since state is to be overwritten, theres no need to keep running the background thread.
    pub fn overwrite<R>(&mut self, f: impl FnOnce(&mut CircuitRuntime) -> R) -> R {
        self.runtime.overwrite(|r| {
            let result = f(r);
            self.is_paused = r.tick_interval.is_none();
            result
        })
    }
}

impl CircuitRuntime {
    pub fn tick_n(&mut self, n: u64) {
        self.engine.tick_n(&mut self.state, n);
    }
}

impl Drop for CircuitRunner {
    fn drop(&mut self) {
        self.runtime.overwrite(|r| r.exit = true);
    }
}
