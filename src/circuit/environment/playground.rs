use super::*;

pub struct CircuitEnvPlayground {
    circuit: CircuitState,
}

impl Clone for CircuitEnvPlayground {
    fn clone(&self) -> Self {
        Self {
            circuit: self.circuit.clone(),
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.circuit.clone_from(&source.circuit);
    }
}

impl CircuitEnvPlayground {
    pub const NAME: &str = "Playground";

    pub fn new(circuit: Arc<CircuitImage>) -> Self {
        Self {
            circuit: CircuitState::new(circuit),
        }
    }
}

impl CircuitEnv for CircuitEnvPlayground {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn circuit(&self) -> &CircuitState {
        &self.circuit
    }

    fn tick(&mut self, engine: &mut dyn CircuitEngine) {
        engine.tick(&mut self.circuit);
    }

    fn tick_n(&mut self, engine: &mut dyn CircuitEngine, n: u64) {
        engine.tick_n(&mut self.circuit, n);
    }

    fn set_input(&mut self, net: u32, powered: bool) {
        self.circuit.nets.inputs_mut()[net as usize] = powered;
    }
}
