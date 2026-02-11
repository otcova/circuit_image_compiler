use super::*;

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct CircuitEnvAdderConfig {
    pub seed: u64,
    pub bits_inp_a: u32,
    pub bits_inp_b: u32,
    pub bits_out: u32,
    pub max_operations: u32,
}

pub struct CircuitEnvAdder {
    config: CircuitEnvAdderConfig,

    circuit: CircuitState,
    rng: SmallRng,
    queue: VecDeque<AdderItem>,

    halt: Option<AdderHalt>,
    operations_done: u32,
}

impl Clone for CircuitEnvAdder {
    fn clone(&self) -> Self {
        Self {
            config: self.config,
            circuit: self.circuit.clone(),
            rng: self.rng.clone(),
            queue: self.queue.clone(),
            halt: self.halt,
            operations_done: self.operations_done,
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.config = source.config;
        self.circuit.clone_from(&source.circuit);
        self.rng.clone_from(&source.rng);
        self.queue.clone_from(&source.queue);
        self.halt = source.halt;
        self.operations_done = source.operations_done;
    }
}

#[derive(Default)]
pub struct EnvAdderIO {
    pub inp_a: u128,
    pub inp_b: u128,

    pub next: bool,
    pub done: bool,
    pub out: u128,
}

#[derive(Clone, Copy)]
pub struct AdderItem {
    pub a: u128,
    pub b: u128,
    pub sum: u128,
}

#[derive(Clone, Copy)]
pub enum AdderHalt {
    Success,
    InvalidIo,
    WrongOut {
        a: u128,
        b: u128,
        expected: u128,
        got: u128,
    },
    UnexpectedOut,
}

impl CircuitEnvAdderConfig {
    pub fn new(circuit: &CircuitImage) -> Self {
        let bits_inp_a = (circuit.inputs().len() as u32 / 2).clamp(1, 128);
        let bits_out = circuit.outputs().len().saturating_sub(2) as u32;

        CircuitEnvAdderConfig {
            seed: 0,
            bits_inp_a,
            bits_inp_b: bits_inp_a,
            bits_out: if bits_out == bits_inp_a + 1 {
                bits_out
            } else {
                bits_inp_a
            },
            max_operations: 100,
        }
    }

    pub fn input_count(&self) -> u32 {
        self.bits_inp_a + self.bits_inp_b
    }

    pub fn output_count(&self) -> u32 {
        2 + self.bits_out
    }

    pub fn inp_a_range(&self) -> std::ops::Range<u32> {
        0..self.bits_inp_a
    }
    pub fn inp_b_range(&self) -> std::ops::Range<u32> {
        self.bits_inp_a..self.bits_inp_a + self.bits_inp_b
    }
    pub fn out_range(&self) -> std::ops::Range<u32> {
        2..2 + self.bits_out
    }
    pub fn next_idx(&self) -> u32 {
        0
    }
    pub fn done_idx(&self) -> u32 {
        1
    }
}

impl CircuitEnvAdder {
    pub const NAME: &str = "Adder";

    pub fn config(&self) -> &CircuitEnvAdderConfig {
        &self.config
    }

    pub fn queue(&self) -> &VecDeque<AdderItem> {
        &self.queue
    }

    pub fn operations_done(&self) -> u32 {
        self.operations_done
    }

    pub fn is_halt(&self) -> Option<AdderHalt> {
        self.halt
    }

    pub fn get_io(&self) -> Result<EnvAdderIO, AdderHalt> {
        if self.circuit.image.inputs().len() != self.config.input_count() as usize
            || self.circuit.image.outputs().len() != self.config.output_count() as usize
        {
            return Err(AdderHalt::InvalidIo);
        }

        Ok(EnvAdderIO {
            inp_a: self.circuit.get_inp_u128(self.config.inp_a_range()),
            inp_b: self.circuit.get_inp_u128(self.config.inp_b_range()),
            out: self.circuit.get_out_u128(self.config.out_range()),
            next: self.circuit.get_out_bool(self.config.next_idx()),
            done: self.circuit.get_out_bool(self.config.done_idx()),
        })
    }

    pub fn new(circuit: Arc<CircuitImage>, config: CircuitEnvAdderConfig) -> Self {
        let mut adder = Self {
            queue: VecDeque::new(),
            halt: None,
            operations_done: 0,
            circuit: CircuitState::new(circuit),

            rng: SmallRng::seed_from_u64(config.seed),
            config,
        };

        adder.halt = adder.get_io().err();
        adder
    }

    fn io_tick(&mut self) {
        if self.halt.is_some() {
            return;
        }

        let io = match self.get_io() {
            Ok(io) => io,
            Err(halt) => {
                self.halt = Some(halt);
                return;
            }
        };

        if io.done {
            // Validate output
            if let Some(item) = self.queue.pop_front() {
                if io.out != item.sum {
                    self.queue.push_front(item);
                    let got = io.out;
                    self.halt = Some(AdderHalt::WrongOut {
                        a: item.a,
                        b: item.b,
                        expected: item.sum,
                        got,
                    });
                    return;
                } else {
                    self.operations_done += 1;
                    if self.operations_done == self.config.max_operations {
                        self.halt = Some(AdderHalt::Success);
                        return;
                    }
                }
            } else {
                // We did not expect output from circuit
                self.halt = Some(AdderHalt::UnexpectedOut);
                return;
            }
        }

        if io.next || self.circuit.tick == 0 {
            // Compute new input
            let a = self.rng.random_range(0..=mask_u128(self.config.bits_inp_a));
            let b = self.rng.random_range(0..=mask_u128(self.config.bits_inp_b));
            let sum = a.wrapping_add(b) & mask_u128(self.config.bits_out);
            self.queue.push_back(AdderItem { a, b, sum });

            // Send new input into the circuit
            self.circuit.set_inp_u128(self.config.inp_a_range(), a);
            self.circuit.set_inp_u128(self.config.inp_b_range(), b);
        }
    }
}

impl CircuitEnv for CircuitEnvAdder {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn circuit(&self) -> &CircuitState {
        &self.circuit
    }

    fn tick(&mut self, engine: &mut dyn CircuitEngine) {
        if self.halt.is_some() {
            return;
        }
        if self.circuit.tick.is_multiple_of(2) {
            self.io_tick();
            if self.halt.is_some() {
                return;
            }
        }
        engine.tick(&mut self.circuit);
    }

    fn set_input(&mut self, net: u32, powered: bool) {
        self.circuit.nets.inputs_mut()[net as usize] = powered;
    }
}
