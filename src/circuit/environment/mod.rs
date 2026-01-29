use super::*;
use rand::{Rng, SeedableRng, rngs::SmallRng};
use std::collections::VecDeque;

pub mod adder;
pub mod collatz;
pub mod playground;

/// The environemnt of a circuit, which consists on:
/// - Circuit net state (wires on/off)
/// - Engine to run the circuit
/// - Logic specific to the environment to interact with the circuit.
pub trait CircuitEnv: Send + 'static {
    fn circuit(&self) -> &CircuitState;

    /// Use tick_n for faster iteration
    fn tick(&mut self, engine: &mut dyn CircuitEngine);

    /// This could be faster than `self.tick` (if n > 1) since engines may:
    /// - Apply optimizations that allow for jumping multiple steps at once.
    /// - Need to transform the state into
    fn tick_n(&mut self, engine: &mut dyn CircuitEngine, n: u64) {
        for _ in 0..n {
            self.tick(engine);
        }
    }

    fn set_input(&mut self, net: u32, powered: bool);

    fn name(&self) -> &'static str;
}

fn mask_u128(size: u32) -> u128 {
    u128::MAX >> (u128::BITS.saturating_sub(size))
}
