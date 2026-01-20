use std::time::{Duration, Instant};

pub fn bench_time<F>(mut f: F, min_time: Duration) -> Duration
where
    F: FnMut(),
{
    let mut samples: Vec<Duration> = Vec::new();
    let bench_start = Instant::now();

    while bench_start.elapsed() < min_time {
        let start = Instant::now();
        f();
        samples.push(start.elapsed());
    }

    assert!(!samples.is_empty(), "benchmark collected no samples");

    samples.sort_unstable();

    let mid = samples.len() / 2;
    if samples.len() % 2 == 0 {
        // Even: average the two middle values
        (samples[mid - 1] + samples[mid]) / 2
    } else {
        samples[mid]
    }
}
