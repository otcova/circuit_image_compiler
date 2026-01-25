use std::{
    ops::{Deref, DerefMut},
    sync::{Condvar, Mutex},
};

/// Shared state container used to synchronize a CPU-bound simulation runner
/// with a synchronous UI thread.
///
/// This design guarantees data-race freedom and bounded lock hold times,
/// while allowing explicit, intentional blocking when required.
pub struct SyncState<S> {
    shared: Mutex<Shared<S>>,
    cv: Condvar,
}

/// Internal shared representation protected by the mutex.
///
/// The `state` and `version` fields are always mutated together while holding
/// the mutex. No access is allowed without synchronization.
struct Shared<S> {
    state: S,
    version: u64,
}

/// Local copy of the simulation state held by the runner.
pub struct Local<S> {
    state: S,
    version: u64,
}

impl<S> SyncState<S> {
    /// Create a new `SyncState` initialized with the given state.
    ///
    /// The initial version is `1`. The threads should initialize its new local state
    /// (which start at version `0`) by pulling from the shared state once.
    pub fn new(initial: S) -> Self {
        Self {
            shared: Mutex::new(Shared {
                state: initial,
                version: 1,
            }),
            cv: Condvar::new(),
        }
    }

    /// Read the latest shared state under a short-lived lock.
    ///
    /// Intended for a quick copy or inspection of the state.
    ///
    /// The mutex is held only for the duration of the closure.
    pub fn get<R>(&self, f: impl FnOnce(&S) -> R) -> R {
        let guard = self.shared.lock().unwrap();
        f(&guard.state)
    }

    /// Lock the shared state exclusively and mutate it.
    ///
    /// This operation:
    /// - blocks the shared state while the lock is held
    /// - increments the shared version
    /// - notifies all waiters on the condition variable
    ///
    /// Intended for:
    /// - holding the state without letting others use it/sync
    /// - operations that do not interleave with the other threads progress
    pub fn overwrite<R>(&self, f: impl FnOnce(&mut S) -> R) -> R {
        let mut guard = self.shared.lock().unwrap();
        let result = f(&mut guard.state);
        guard.version += 1;
        self.cv.notify_all();
        result
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SyncOutcome {
    Published,
    Overwritten,
}

#[allow(dead_code)]
impl<S: Clone> SyncState<S> {
    /// Block the current thread while `cond(&state)` evaluates to `true`.
    ///
    /// This method:
    /// - does not consume CPU while waiting
    /// - handles spurious wakeups correctly
    /// - reevaluates the condition under the mutex
    ///
    /// Once the condition becomes false, local and shared states are synced.
    ///
    /// Intended to pause a thread while configuration says so.
    pub fn wait_while(
        &self,
        local: &mut Local<S>,
        mut cond: impl FnMut(&S) -> bool,
    ) -> SyncOutcome {
        let mut guard = self.shared.lock().unwrap();
        while cond(&guard.state) {
            guard = self.cv.wait(guard).unwrap();
        }
        self.sync(&mut *guard, local)
    }

    /// Synchronize a thread local state with the shared state.
    ///
    /// # Semantics
    ///
    /// - If `local.version == shared.version`:
    ///   - The local state is copied into the shared state and shared version is incremented
    /// - Otherwise:
    ///   - The shared state is copied into the local state
    ///
    /// In both cases, the local version is updated to match the shared version.
    ///
    /// All state and version updates happen while holding the mutex.
    pub fn sync_state(&self, local: &mut Local<S>) -> SyncOutcome {
        let mut shared = self.shared.lock().unwrap();
        self.sync(&mut *shared, local)
    }

    fn sync(&self, shared: &mut Shared<S>, local: &mut Local<S>) -> SyncOutcome {
        if local.version == shared.version {
            // Runner publishes
            shared.state.clone_from(&local.state);
            shared.version += 1;
            local.version = shared.version;
            self.cv.notify_all();
            SyncOutcome::Published
        } else {
            // UI overwrote shared state
            local.state.clone_from(&shared.state);
            local.version = shared.version;
            SyncOutcome::Overwritten
        }
    }
}

impl<S> Local<S> {
    // Initializes a local state at version `0`
    pub fn new(state: S) -> Self {
        Self { state, version: 0 }
    }
}

impl<S> Deref for Local<S> {
    type Target = S;
    fn deref(&self) -> &S {
        &self.state
    }
}

impl<S> DerefMut for Local<S> {
    fn deref_mut(&mut self) -> &mut S {
        &mut self.state
    }
}
