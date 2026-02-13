use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::circuit::environment::CircuitEnv;
use crate::utils::sync_state::*;
use crate::{circuit::*, utils::clock::Clock};
use std::{sync::Arc, thread, time::Duration};

pub struct CircuitRuntime<E: CircuitEnv> {
    pub engine: Box<dyn CircuitEngine>,
    pub env: E,
    /// Delta time between circuit ticks.
    pub tick_interval: Duration,
    pub paused: bool,
    exit: bool,
}

impl<E: CircuitEnv + Clone> Clone for CircuitRuntime<E> {
    fn clone(&self) -> Self {
        Self {
            engine: self.engine.clone_dyn(),
            env: self.env.clone(),
            tick_interval: self.tick_interval,
            paused: self.paused,
            exit: self.exit,
        }
    }

    fn clone_from(&mut self, other: &Self) {
        if self.engine.name() != other.engine.name() {
            self.engine = other.engine.clone_dyn();
        }
        self.env.clone_from(&other.env);
        self.tick_interval = other.tick_interval;
        self.paused = other.paused;
        self.exit = other.exit;
    }
}

pub trait CircuitRunnerTrait {
    fn env(&self) -> &dyn CircuitEnv;
    fn env_mut(&mut self) -> &mut dyn CircuitEnv;
    fn update(&mut self) -> SyncOutcome;
    fn publish(&mut self) -> SyncOutcome;
    fn circuit(&self) -> &CircuitState;

    fn engine(&self) -> &dyn CircuitEngine;
    fn set_engine(&mut self, engine: Box<dyn CircuitEngine>);

    fn tick_interval(&self) -> Duration;
    fn set_tick_interval(&mut self, interval: Duration);
    fn set_tick_interval_secs(&mut self, interval_secs: f32) {
        self.set_tick_interval(Duration::from_secs_f32(interval_secs));
    }

    fn is_paused(&self) -> bool;
    fn set_paused(&mut self, paused: bool);

    fn tick_n(&mut self, n: u64);
}

pub struct CircuitRunner<E: CircuitEnv + Clone> {
    // Runtime thread will only block the mutex quickly to update the state with memcpy,
    // wich will happen at most every few ms.
    // Try Optimize: This might be better using tokio::watch instead of Mutex.
    sync: Arc<SyncState<CircuitRuntime<E>>>,

    pub runtime: Local<CircuitRuntime<E>>,
}

impl<Env: CircuitEnv + Clone> CircuitRunner<Env> {
    /// How often the runner thread should update global state.
    /// 8ms corresponds to about twice per frame in 60fps.
    const UPDATE_INTERVAL: Duration = Duration::from_millis(8);

    pub fn new(env: Env, engine: Box<dyn CircuitEngine>) -> Self {
        let sync_runtime = Arc::new(SyncState::new(CircuitRuntime {
            engine,
            env,
            tick_interval: Duration::ZERO,
            paused: true,
            exit: false,
        }));
        let sync_runtime_clone = sync_runtime.clone();

        thread::spawn(move || {
            let mut runtime = sync_runtime.new_local();
            let mut clock = Clock::new(Self::UPDATE_INTERVAL);

            // We introduce some randomness to the amount of steps advanced to prevent
            // the sampling frequency to align with the nets frequencies.
            let mut rng = SmallRng::seed_from_u64(0);

            while !runtime.exit {
                // Run ticks and sleep thread for UPDATE_INTERVAL time
                if !runtime.paused {
                    // Do at most 256 at a time to not exceed UPDATE_INTERVAL significatively.
                    let max_ticks_per_step = rng.random_range(128_u32..256);

                    clock.run_ticks(runtime.tick_interval, |n| {
                        let n = n.min(max_ticks_per_step);
                        runtime.tick_n(n as u64);
                        n
                    });
                }

                let mut reset_clock = false;
                let prev_tick_interval = runtime.tick_interval;

                sync_runtime.wait_while(&mut runtime, |s| {
                    let wait = !s.exit && s.paused;
                    if s.paused || s.tick_interval != prev_tick_interval {
                        reset_clock = true;
                    }
                    wait
                });

                if reset_clock {
                    clock.reset();
                }
            }
        });

        Self {
            runtime: sync_runtime_clone.new_local(),
            sync: sync_runtime_clone,
        }
    }
}

impl<E: CircuitEnv + Clone> CircuitRunnerTrait for CircuitRunner<E> {
    fn env(&self) -> &dyn CircuitEnv {
        &self.runtime.env
    }

    fn env_mut(&mut self) -> &mut dyn CircuitEnv {
        &mut self.runtime.env
    }

    fn update(&mut self) -> SyncOutcome {
        self.sync.sync(&mut self.runtime)
    }

    // Publish local runtime into the runner thread
    fn publish(&mut self) -> SyncOutcome {
        self.sync.publish(&mut self.runtime)
    }

    fn circuit(&self) -> &CircuitState {
        self.runtime.env.circuit()
    }

    fn engine(&self) -> &dyn CircuitEngine {
        &*self.runtime.engine
    }

    fn set_engine(&mut self, engine: Box<dyn CircuitEngine>) {
        self.runtime.engine = engine;
    }

    fn tick_interval(&self) -> Duration {
        self.runtime.tick_interval
    }

    fn set_tick_interval(&mut self, interval: Duration) {
        self.runtime.tick_interval = interval;
    }

    fn is_paused(&self) -> bool {
        self.runtime.paused
    }

    fn set_paused(&mut self, paused: bool) {
        self.runtime.paused = paused;
    }

    fn tick_n(&mut self, n: u64) {
        self.runtime.tick_n(n);
    }
}

impl<E: CircuitEnv> CircuitRuntime<E> {
    pub fn tick_n(&mut self, n: u64) {
        self.env.tick_n(&mut *self.engine, n);
    }
}

impl<E: CircuitEnv + Clone> Drop for CircuitRunner<E> {
    fn drop(&mut self) {
        self.sync.mut_shared(|r| r.exit = true);
    }
}
