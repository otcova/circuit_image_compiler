use std::cell::UnsafeCell;
use tokio::sync::oneshot;

pub use oneshot::Sender;

/// Internal state tracking whether the value is pending or available.
enum InnerPromise<T> {
    Loading(oneshot::Receiver<T>),
    Done(T),
}

impl<T> InnerPromise<T> {
    /// Checks the receiver. If data is ready, updates state to `Done` and returns the value.
    fn poll(&mut self) -> Option<&mut T> {
        match self {
            InnerPromise::Done(val) => Some(val),
            InnerPromise::Loading(rx) => match rx.try_recv() {
                Err(_) => None,
                Ok(val) => {
                    *self = InnerPromise::Done(val);
                    match self {
                        InnerPromise::Done(val) => Some(val),
                        InnerPromise::Loading(_) => None,
                    }
                }
            },
        }
    }
}

/// A wrapper around a oneshot channel that stores the result once received.
/// Note: This type is `!Sync`. It cannot be shared across threads.
pub struct Promise<T>(UnsafeCell<InnerPromise<T>>);

impl<T> Promise<T> {
    /// Creates a new `Promise` and its corresponding `Sender`.
    pub fn new() -> (Self, Sender<T>) {
        let (tx, rx) = oneshot::channel();
        let loading = UnsafeCell::new(InnerPromise::Loading(rx));
        (Promise(loading), tx)
    }

    /// Tries to get the value. If the value has arrived, the internal state
    /// updates and a reference is returned.
    pub fn get(&self) -> Option<&T> {
        // SAFETY:
        // 1. `UnsafeCell` is `!Sync`, ensuring `get` cannot be called concurrently
        //    from multiple threads, preventing data races.
        // 2. We are not calling user code (callbacks) inside the unsafe block,
        //    preventing re-entrancy issues where `get` calls `get` recursively
        //    while the mutable reference is active.
        // 3. We downgrade the `&mut T` returned by `poll` to `&T`, respecting
        //    Rust's borrowing rules for the public API.
        unsafe { (&mut *self.0.get()).poll().map(|x| &*x) }
    }

    /// Returns a mutable reference to the value if available.
    /// Since we have `&mut self`, no unsafe is needed.
    pub fn get_mut(&mut self) -> Option<&mut T> {
        self.0.get_mut().poll()
    }

    /// Checks if the promise is done without consuming it.
    /// Equivalent to: `self.get().is_some()`
    pub fn is_done(&self) -> bool {
        self.get().is_some()
    }
}
