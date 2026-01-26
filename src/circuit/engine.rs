use crate::utils::{bench::bench_seconds, union_find::UnionFind};

use super::*;
use std::time::Duration;

pub fn default_engine(_circuit: &CircuitImage) -> impl CircuitEngine + 'static {
    CircuitEngineDfs::default()
}

pub fn all_engines(_circuit: &CircuitImage) -> SmallVec<Box<dyn CircuitEngine>, 16> {
    SmallVec::from_buf([
        Box::new(CircuitEngineDfs::default()),
        Box::new(CircuitEngineUf::default()),
    ])
}

#[derive(Clone, Copy)]
pub struct EngineBenchmarkResult {
    pub engine_name: &'static str,
    pub tps: f32,
}

/// Responisble of executing a circuit.
pub trait CircuitEngine: Send {
    fn name(&self) -> &'static str;

    fn clone_dyn(&self) -> Box<dyn CircuitEngine>;

    /// Crates a new engine for another circuit with the configuration of Self.
    fn new_dyn(&self, circuit: &CircuitImage) -> Box<dyn CircuitEngine> {
        // Most engines do not depend on the circuit
        let _ = circuit;

        self.clone_dyn()
    }

    /// Use step_n for faster iteration
    fn tick(&mut self, state: &mut CircuitState);

    /// This may be faster than step_n (if n > 1) since engines may:
    /// - Apply optimizations that allow for jumping multiple steps at once.
    /// - Need to transform the state into
    fn tick_n(&mut self, state: &mut CircuitState, n: u64) {
        for _ in 0..n {
            self.tick(state);
        }
    }

    /// ticks per second mesured after running for the amount of time specified.
    fn bench_tps(&mut self, state: &mut CircuitState, min_time: Duration) -> EngineBenchmarkResult {
        let step_by_n: u64 = 32;
        let time = bench_seconds(state, |s| self.tick_n(s, step_by_n), min_time);
        EngineBenchmarkResult {
            engine_name: self.name(),
            tps: step_by_n as f32 / time,
        }
    }
}

/// Stores both the nets on/off state (deref to [bool]),
/// and the input on/off state of the same nets.
#[derive(Clone)]
pub struct CircuitStateNets(pub Box<[bool]>);

#[allow(dead_code)]
impl CircuitStateNets {
    pub fn new(circuit: &CircuitImage) -> Self {
        let mut nets = Self(vec![false; circuit.net_count() as usize * 2].into_boxed_slice());
        if let Some(power_net) = nets.inputs_mut().get_mut(NET_ON as usize) {
            *power_net = true;
        }
        nets
    }
    pub fn reset(&mut self) {
        self.0.fill(false);
    }
    pub fn get(&self) -> &[bool] {
        &self.0[..self.0.len() / 2]
    }
    pub fn get_mut(&mut self) -> &mut [bool] {
        let nets = self.0.len() / 2;
        &mut self.0[..nets]
    }
    pub fn nets_and_inputs_mut(&mut self) -> (&mut [bool], &mut [bool]) {
        let nets = self.0.len() / 2;
        self.0.split_at_mut(nets)
    }
    pub fn inputs(&self) -> &[bool] {
        &self.0[self.0.len() / 2..]
    }
    pub fn inputs_mut(&mut self) -> &mut [bool] {
        let nets = self.0.len() / 2;
        &mut self.0[nets..]
    }

    /// Returns the on/off state of nets and inputs in a single slice
    /// This is basically how its stored internally.
    pub fn as_concat(&self) -> &[bool] {
        &self.0
    }
}

impl CircuitState {
    /// Given the state of the wires, compute the state of the gates.
    /// Since this marks the end of a tick, this function also increments the `tick` counter
    pub fn update_gates(&mut self) {
        let nets = self.nets.get_mut();
        for (gate_i, gate) in self.image.gates.iter().enumerate() {
            let gate_net = gate_i + self.image.wire_count() as usize;
            nets[gate_net] = gate.wires.iter().any(|&net| nets[net as usize]);
        }
        self.tick += 1;
    }

    fn apply_inputs(&mut self) {
        let wires = self.image.wire_count() as usize;
        let (nets, inputs) = self.nets.nets_and_inputs_mut();
        nets[..wires].copy_from_slice(&inputs[..wires]);
    }
}

/// Implements:
/// - UnionFind
#[derive(Default)]
pub struct CircuitEngineUf {
    wire_connections: UnionFind,
}

impl CircuitEngine for CircuitEngineUf {
    fn name(&self) -> &'static str {
        "Union-Find"
    }

    fn clone_dyn(&self) -> Box<dyn CircuitEngine> {
        Box::new(Self::default())
    }

    /// Perfoms a single tick. Equivalent to `update_wires()` + `update_gates()`
    fn tick(&mut self, state: &mut CircuitState) {
        if state.tick.is_multiple_of(2) {
            self.update_wires(state);
            state.tick += 1;
        } else {
            state.update_gates();
        }
    }
}

impl CircuitEngineUf {
    /// Given the state of the gates, compute the state of the wires.
    pub fn update_wires(&mut self, state: &mut CircuitState) {
        self.wire_connections.clear();
        self.wire_connections.extend(state.image.wire_count());

        // TODO: init here with push instead of extend
        for (net, &is_powered) in state.nets.inputs().iter().enumerate() {
            if is_powered {
                self.wire_connections.set_parent_unchecked(net as u32, 0);
            }
        }

        let nets = state.nets.get_mut();

        for buffer_gate in &state.image.buffer_gates {
            let toggled = buffer_gate.controls.iter().all(|&net| nets[net as usize]);
            if GateType::Passive.connects_wires(toggled) {
                for &net in &buffer_gate.wires {
                    self.wire_connections.set_parent_unchecked(net, 0);
                }
            }
        }

        for not_gate in &state.image.not_gates {
            let toggled = not_gate.controls.iter().all(|&net| nets[net as usize]);
            if GateType::Active.connects_wires(toggled) {
                for &net in &not_gate.wires {
                    self.wire_connections.set_parent_unchecked(net, 0);
                }
            }
        }

        // Connect nets from gates
        for gate in &state.image.non_trivial_gates {
            let toggled = gate.controls.iter().all(|&net| nets[net as usize]);
            if gate.ty.connects_wires(toggled) {
                for net in gate.wires.windows(2) {
                    self.wire_connections.alias(net[0], net[1]);
                }
            }
        }

        // Write wire state
        for wire_net in 1..state.image.wire_count() {
            // Since we are iterating in order, and NET_ON is lower than the visited,
            // we can use grandparent instead of root.
            nets[wire_net as usize] = self.wire_connections.has_grandparent(wire_net, 0);
        }
    }
}

/// Implements:
/// - Depth First Search
#[derive(Default)]
pub struct CircuitEngineDfs {
    dfs_stack: Vec<u32>,

    /// Weather a certain gate is or not connection its wires.
    non_trivial_is_connected: Vec<bool>,
}

impl CircuitEngine for CircuitEngineDfs {
    fn name(&self) -> &'static str {
        "DFS"
    }

    fn clone_dyn(&self) -> Box<dyn CircuitEngine> {
        Box::new(Self::default())
    }

    /// Perfoms a single step. Equivalent to `update_wires()` + `update_gates()`
    fn tick(&mut self, state: &mut CircuitState) {
        if state.tick.is_multiple_of(2) {
            self.update_wires(state);
            state.tick += 1;
        } else {
            state.update_gates();
        }
    }
}

impl CircuitEngineDfs {
    /// Given the state of the gates, compute the state of the wires.
    pub fn update_wires(&mut self, state: &mut CircuitState) {
        // state.nets[NET_ON as usize] = true;
        // state.nets[2..state.circuit.wire_count() as usize].fill(false);
        state.apply_inputs();
        let nets = state.nets.get_mut();

        for buffer_gate in &state.image.buffer_gates {
            let toggled = buffer_gate.controls.iter().all(|&net| nets[net as usize]);
            if GateType::Passive.connects_wires(toggled) {
                for &net in &buffer_gate.wires {
                    nets[net as usize] = true;
                }
            }
        }

        for not_gate in &state.image.not_gates {
            let toggled = not_gate.controls.iter().all(|&net| nets[net as usize]);
            if GateType::Active.connects_wires(toggled) {
                for &net in &not_gate.wires {
                    nets[net as usize] = true;
                }
            }
        }

        self.non_trivial_is_connected.clear();
        for gate in &state.image.non_trivial_gates {
            let is_toggled = gate.controls.iter().all(|&net| nets[net as usize]);
            let is_connected = gate.ty.connects_wires(is_toggled);
            self.non_trivial_is_connected.push(is_connected);
        }

        self.dfs_stack.clear();
        for root_wire in NET_ON..state.image.wire_count() {
            // We use state as visited map for the dfs
            if !nets[root_wire as usize] {
                continue;
            }

            self.dfs_stack.push(root_wire);

            // For all queued nodes (wires)
            while let Some(wire) = self.dfs_stack.pop() {
                // Visit all edges (gates)
                for &gate_idx in &state.image.wires_non_trivial[wire as usize] {
                    // If edge is enabled (gate connects wires)
                    if self.non_trivial_is_connected[gate_idx as usize] {
                        self.non_trivial_is_connected[gate_idx as usize] = true;
                        // Visit the neighbours
                        for &neighbour in &state.image.non_trivial_gates[gate_idx as usize].wires {
                            if !nets[neighbour as usize] {
                                self.dfs_stack.push(neighbour);
                                nets[neighbour as usize] = true;
                            }
                        }
                    }
                }
            }
        }
    }
}
