use std::{
    ops::{Deref, DerefMut},
    sync::{Condvar, Mutex},
};

/// Shared state container used to synchronize a CPU-bound simulation runner
/// with a synchronous UI thread.
///
/// This design guarantees data-race freedom and bounded lock hold times,
/// while allowing explicit, intentional blocking when required.
pub struct SyncState<S: Clone> {
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
#[derive(Clone)]
pub struct Local<S> {
    state: S,
    version: u64,

    // Did local state mutate since last synchronization
    mutated: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SyncOutcome {
    Published,
    Overwritten,
    NoChanges,
}

#[allow(dead_code)]
impl<S: Clone> SyncState<S> {
    pub fn new(initial: S) -> Self {
        Self {
            shared: Mutex::new(Shared {
                state: initial,
                version: 1,
            }),
            cv: Condvar::new(),
        }
    }

    pub fn mut_shared<R>(&self, mut f: impl FnMut(&mut S) -> R) -> R {
        let mut shared = self.shared.lock().unwrap_or_else(|p| p.into_inner());
        let result = f(&mut shared.state);
        shared.version += 1;
        self.cv.notify_all();
        result
    }

    pub fn publish_new_local(&self, state: S) -> Local<S> {
        let mut shared = self.shared.lock().unwrap_or_else(|p| p.into_inner());
        shared.publish_state(state, &self.cv);
        shared.new_local()
    }

    pub fn new_local(&self) -> Local<S> {
        let shared = self.shared.lock().unwrap_or_else(|p| p.into_inner());
        shared.new_local()
    }

    pub fn publish(&self, local: &mut Local<S>) -> SyncOutcome {
        let mut shared = self.shared.lock().unwrap_or_else(|p| p.into_inner());
        shared.publish(local, &self.cv)
    }

    pub fn overwrite_local(&self, local: &mut Local<S>) -> SyncOutcome {
        let mut shared = self.shared.lock().unwrap_or_else(|p| p.into_inner());
        shared.overwrite_local(local)
    }

    pub fn wait_while(
        &self,
        local: &mut Local<S>,
        mut cond: impl FnMut(&S) -> bool,
    ) -> SyncOutcome {
        let mut shared = self.shared.lock().unwrap_or_else(|p| p.into_inner());

        if local.version == shared.version {
            shared.publish(local, &self.cv);
        }

        while cond(&shared.state) {
            shared = self.cv.wait(shared).unwrap_or_else(|p| p.into_inner());
        }

        if local.version == shared.version {
            // "overwrite_shared" already done
            SyncOutcome::Published
        } else {
            shared.overwrite_local(local);
            SyncOutcome::Overwritten
        }
    }

    pub fn sync(&self, local: &mut Local<S>) -> SyncOutcome {
        match self.shared.lock() {
            Ok(mut shared) => shared.sync(local, &self.cv),
            Err(mut poisoned) => {
                poisoned.get_mut().publish(local, &self.cv);
                SyncOutcome::Overwritten
            }
        }
    }
}

impl<S: Clone> Shared<S> {
    fn sync(&mut self, local: &mut Local<S>, cv: &Condvar) -> SyncOutcome {
        if local.version == self.version {
            self.publish(local, cv)
        } else {
            self.overwrite_local(local)
        }
    }

    fn publish(&mut self, local: &mut Local<S>, cv: &Condvar) -> SyncOutcome {
        if local.version == self.version && !local.mutated {
            return SyncOutcome::NoChanges;
        }

        self.state.clone_from(&local.state);
        self.version = self.version.wrapping_add(1);
        local.version = self.version;
        local.mutated = false;
        cv.notify_all();
        SyncOutcome::Published
    }

    fn overwrite_local(&mut self, local: &mut Local<S>) -> SyncOutcome {
        if local.version == self.version && !local.mutated {
            return SyncOutcome::NoChanges;
        }

        local.state.clone_from(&self.state);
        local.version = self.version;
        local.mutated = false;
        SyncOutcome::Overwritten
    }

    fn publish_state(&mut self, state: S, cv: &Condvar) {
        self.state = state;
        self.version = self.version.wrapping_add(1);
        cv.notify_all();
    }

    fn new_local(&self) -> Local<S> {
        Local {
            state: self.state.clone(),
            version: self.version,
            mutated: false,
        }
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
        self.mutated = true;
        &mut self.state
    }
}
