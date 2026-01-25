use std::time::{Duration, Instant};

/// A catch-up clock that runs ticks at a requested interval.
///
/// The clock does **not** sleep between ticks. Instead, during each
/// `report_interval` iteration it:
///
/// 1. Runs all ticks that are due based on `tick_interval`
/// 2. Sleeps only once, to complete the `report_interval`
///
/// If the CPU cannot keep up, ticks are executed slower
/// (best-effort, max-CPU behavior).
pub struct Clock {
    /// Duration of a single reporting cycle.
    report_interval: Duration,

    /// Instant at which the next tick is due.
    next_tick: Instant,
}

impl Clock {
    /// Creates a new `Clock`.
    ///
    /// `report_interval` controls how often `run_iteration` completes.
    pub fn new(report_interval: Duration) -> Self {
        Self {
            report_interval,
            next_tick: Instant::now(),
        }
    }

    // Sets next tick to be done now.
    // Usefull after pausing clock.
    pub fn reset(&mut self) {
        self.next_tick = Instant::now();
    }

    /// Runs one reporting iteration.
    ///
    /// # Parameters
    /// - `tick_interval`: Desired interval between ticks for this iteration.
    ///   This value may change between calls.
    /// - `tick`: Function invoked once per tick.
    ///
    /// # Behavior
    /// - Executes all ticks that are due since the previous iteration
    /// - Does **not** sleep between ticks
    /// - Sleeps only to complete the `report_interval`
    ///
    /// If execution falls behind, ticks are executed as fast as possible
    /// until reaching the `report_interval` limit.
    pub fn run_ticks<F>(&mut self, tick_interval: Duration, mut ticks: F)
    where
        F: FnMut(u32) -> u32,
    {
        let start = Instant::now();

        // Run all ticks that are due
        if self.next_tick <= start {
            let overdue = start - self.next_tick;
            // Number of ticks to run (at least 1)
            let ticks_due = if tick_interval.as_nanos() == 0 {
                u32::MAX
            } else {
                let ticks_due = 1 + (overdue.as_nanos() / tick_interval.as_nanos());
                ticks_due.try_into().unwrap_or(u32::MAX)
            };

            let mut ticks_done = 0;
            while ticks_done < ticks_due {
                ticks_done += ticks(ticks_due - ticks_done);

                let now = Instant::now();
                if now - start > self.report_interval {
                    // Case where tick does not run fast enough
                    // We restart next_tick to not accumulate undoable work.
                    self.next_tick = now;
                    return;
                }
            }

            self.next_tick += tick_interval.saturating_mul(ticks_done);
        }

        // Sleep to finish the reporting interval
        let elapsed = start.elapsed();
        if elapsed < self.report_interval {
            std::thread::sleep(self.report_interval - elapsed);
        }
    }
}
