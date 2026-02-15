use std::{
    hint::black_box,
    time::{Duration, Instant},
};

use cpu_time::ProcessTime;

struct FallbackInstant {
    global: Instant,
    cpu: Option<ProcessTime>,
}

impl FallbackInstant {
    pub fn now() -> Self {
        Self {
            global: Instant::now(),
            cpu: ProcessTime::try_now().ok(),
        }
    }
    pub fn elapsed(&self) -> f64 {
        if let Some(cpu) = &self.cpu
            && let Ok(cpu_dt) = cpu.try_elapsed()
        {
            // CPU time should be at most global time.
            cpu_dt
                .as_secs_f64()
                .min(self.global.elapsed().as_secs_f64())
        } else {
            self.global.elapsed().as_secs_f64()
        }
    }
}

pub fn bench_seconds<I, R, F>(input: &mut I, mut f: F, min_time: Duration) -> f32
where
    F: FnMut(&mut I) -> R,
{
    let mut samples = Vec::new();
    let bench_start = FallbackInstant::now();

    let start = FallbackInstant::now();
    black_box(f(black_box(input)));
    samples.push(start.elapsed());

    while bench_start.elapsed() < min_time.as_secs_f64() {
        let start = FallbackInstant::now();
        black_box(f(black_box(input)));
        samples.push(start.elapsed());
    }

    if samples.is_empty() {
        return f32::NAN;
    }

    // --- Discard first iterations ---
    const WARMUP_ITERATIONS: usize = 64;
    const MAX_WARMUP_RATIO: usize = 16; // max of 1 warmup per every 16 samples

    let warmup = WARMUP_ITERATIONS.min(samples.len() / MAX_WARMUP_RATIO);
    let samples = &mut samples[warmup..];

    // --- Discard qunatiles ---
    const KEEP: f32 = 0.1; // keep 10%, remove first 45% and last 45% samples
    const MIN_SAMPLES: usize = 3; // At least keep 3 samples

    let keep_count = MIN_SAMPLES.max((samples.len() as f32 * KEEP).round() as usize);
    let keep_count = keep_count.min(samples.len());
    let trim_count = (samples.len() - keep_count) / 2;
    let keep_count = samples.len() - trim_count * 2; // In case trim_count is not divisible by 2

    if trim_count == 0 {
        return (samples.iter().sum::<f64>() / samples.len() as f64) as f32;
    }

    let (_, _, samples) = samples.select_nth_unstable_by(trim_count - 1, |a, b| a.total_cmp(b));
    let (samples, _, _) = samples.select_nth_unstable_by(keep_count, |a, b| a.total_cmp(b));

    // --- Mean remaining samples ---
    (samples.iter().sum::<f64>() / samples.len() as f64) as f32
}
