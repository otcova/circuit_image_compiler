use crate::bench::bench_seconds;

use super::*;
use std::{
    ops::{Deref, DerefMut},
    time::Duration,
};

pub fn default_engine(_circuit: &CircuitImage) -> impl CircuitEngine + 'static {
    CircuitEngineUfT::default()
}

/// Responisble of executing a circuit.
pub trait CircuitEngine {
    fn name(&self) -> &'static str;

    /// Crates a new engine with the configuration of this one for the given circuit.
    fn new_dyn(&self, circuit: &CircuitImage) -> Box<dyn CircuitEngine>;

    /// Use step_n for faster iteration
    fn step(&mut self, circuit: &CircuitImage, state: &mut CircuitStateNets);

    /// This is faster than step_n (if n > 1) since engines may:
    /// - Apply optimizations that allow for jumping multiple steps at once.
    /// - Need to transform the state into
    fn step_n(&mut self, circuit: &CircuitImage, state: &mut CircuitStateNets, n: u64) {
        for _ in 0..n {
            self.step(circuit, state);
        }
    }

    /// Runs a steps for the amount of time specified.
    /// Returns the steps/second.
    fn bench(&mut self, circuit: &CircuitImage, min_time: Duration) -> f32 {
        let step_by_n: u64 = 32;
        let state = &mut CircuitStateNets::new(circuit);
        let time = bench_seconds(state, |s| self.step_n(circuit, s, step_by_n), min_time);
        step_by_n as f32 / time
    }
}

pub struct CircuitStateNets(pub Box<[bool]>);

impl Deref for CircuitStateNets {
    type Target = [bool];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for CircuitStateNets {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl CircuitStateNets {
    pub fn new(circuit: &CircuitImage) -> Self {
        Self(vec![false; circuit.net_count() as usize].into_boxed_slice())
    }
    pub fn reset(&mut self) {
        self.fill(false);
    }
}

impl CircuitStateNets {
    /// Given the state of the wires, compute the state of the gates.
    pub fn update_gates(&mut self, circuit: &CircuitImage) {
        debug_assert!(circuit.net_count() as usize == self.len());
        for (gate_i, gate) in circuit.gates.iter().enumerate() {
            let gate_net = gate_i + circuit.wire_count() as usize;
            self[gate_net] = gate.wires.iter().any(|&net| self[net as usize]);
        }
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
        "UnionFind"
    }

    fn new_dyn(&self, _: &CircuitImage) -> Box<dyn CircuitEngine> {
        Box::new(Self::default())
    }

    /// Perfoms a single step. Equivalent to `update_wires()` + `update_gates()`
    fn step(&mut self, circuit: &CircuitImage, net_state: &mut CircuitStateNets) {
        self.update_wires(circuit, net_state);
        net_state.update_gates(circuit);
    }
}

impl CircuitEngineUf {
    /// Given the state of the gates, compute the state of the wires.
    pub fn update_wires(&mut self, circuit: &CircuitImage, state: &mut [bool]) {
        debug_assert!(circuit.net_count() as usize == state.len());
        state[NET_ON as usize] = true;

        self.wire_connections.clear();
        self.wire_connections.extend(circuit.wire_count());

        // Connect nets from gates
        for gate in &circuit.gates {
            let toggled = gate.controls.iter().all(|&net| state[net as usize]);
            if gate.ty.connects_wires(toggled) {
                for net in gate.wires.windows(2) {
                    self.wire_connections.alias(net[0], net[1]);
                }
            }
        }

        // Write wire state
        for wire_net in 2..circuit.wire_count() {
            // Since we are iterating in order, and NET_ON is lower than the visited,
            // we can use grandparent instead of root.
            state[wire_net as usize] = self.wire_connections.has_grandparent(wire_net, NET_ON);
        }
    }
}

/// Implements:
/// - Depth First Search
///
/// Wires are nodes, Gates are also nodes
/// A wires only connect to gates, and gate only to wires.
///
/// The stack Vec is used as a stack, not as a simple todo list of nets to visit.
#[derive(Default)]
pub struct CircuitEngineMadDfs {
    // (net, gates visited, neighbours visited)
    dfs_stack: Vec<(u32, u32, u32)>,

    /// Weather a certain gate is or not connection its wires.
    connection_state: Vec<bool>,
}

impl CircuitEngine for CircuitEngineMadDfs {
    fn name(&self) -> &'static str {
        "Mad DFS"
    }

    fn new_dyn(&self, _: &CircuitImage) -> Box<dyn CircuitEngine> {
        Box::new(Self::default())
    }

    /// Perfoms a single step. Equivalent to `update_wires()` + `update_gates()`
    fn step(&mut self, circuit: &CircuitImage, net_state: &mut CircuitStateNets) {
        self.update_wires(circuit, net_state);
        net_state.update_gates(circuit);
    }
}

impl CircuitEngineMadDfs {
    /// Given the state of the gates, compute the state of the wires.
    pub fn update_wires(&mut self, circuit: &CircuitImage, state: &mut [bool]) {
        debug_assert!(circuit.net_count() as usize == state.len());

        /// This removes a loot of `as usize`
        const fn s(net: u32) -> usize {
            net as usize
        }

        // 1. state used as visited map for wire nodes
        // 2. self.connection_state used as NOT visited map for gate nodes

        // Init NOT visited map for gates
        self.connection_state.clear();
        for gate in &circuit.gates {
            let toggled = gate.controls.iter().all(|&net| state[net as usize]);
            self.connection_state.push(gate.ty.connects_wires(toggled));
        }

        // Reset visited map for wires
        state[2..circuit.wire_count() as usize].fill(false);
        state[NET_ON as usize] = true;

        self.dfs_stack.clear();

        // Find first node
        let wire = NET_ON;
        let gates = &circuit.wires[s(wire)];

        for (next_visited_gates, &next_gate_idx) in gates.iter().enumerate() {
            if self.connection_state[s(next_gate_idx)] {
                self.connection_state[s(next_gate_idx)] = false;
                let next_gate = &circuit.gates[s(next_gate_idx)];
                let mut wire_found = false;
                for (next_wire_visited, &next_wire) in next_gate.wires.iter().enumerate() {
                    if !state[next_wire as usize] {
                        state[next_wire as usize] = true;
                        self.dfs_stack.push((
                            wire,
                            next_visited_gates as u32,
                            next_wire_visited as u32,
                        ));
                        wire_found = true;
                        break;
                    }
                }
                if wire_found {
                    break;
                }
            }
        }

        // For all queued nodes (wires)
        while let Some((parent_wire, visited_gates, visited_wires)) = self.dfs_stack.pop() {
            let parent_gates = &circuit.wires[s(parent_wire)];
            let gate = {
                let gate_idx = parent_gates[s(visited_gates)];
                &circuit.gates[s(gate_idx)]
            };
            let wire = gate.wires[s(visited_wires)];
            let gates = &circuit.wires[s(wire)];

            // println!(
            //     "<parent_wire: {}, gate: {} ({}/{}), wire: {} ({}/{}) > {:?}",
            //     parent_wire,
            //     circuit.wire_count() + parent_gates[s(visited_gates)],
            //     visited_gates + 1,
            //     parent_gates.len(),
            //     wire,
            //     visited_wires + 1,
            //     gate.wires.len(),
            //     &self.dfs_stack
            // );

            let mut gate_finshed = true;

            // Push next wire in the gate of parent_wire
            for next_wire_visited in visited_wires + 1..gate.wires.len() as u32 {
                let next_wire = gate.wires[s(next_wire_visited)];
                if !state[s(next_wire)] {
                    state[s(next_wire)] = true;
                    self.dfs_stack
                        .push((parent_wire, visited_gates, next_wire_visited));

                    // Gate finished only if this is the last wire of the gate.
                    gate_finshed = next_wire_visited == gate.wires.len() as u32 - 1;
                    break;
                }
            }

            // Push next gate (and next wire from next gate) of parent_wire (if current gate has been done)
            if gate_finshed {
                for next_visited_gates in visited_gates + 1..parent_gates.len() as u32 {
                    let next_gate_idx = parent_gates[s(next_visited_gates)];

                    if self.connection_state[s(next_gate_idx)] {
                        self.connection_state[s(next_gate_idx)] = false;
                        let next_gate = &circuit.gates[s(next_gate_idx)];
                        let mut wire_found = false;
                        for (next_wire_visited, &next_wire) in next_gate.wires.iter().enumerate() {
                            if !state[next_wire as usize] {
                                state[next_wire as usize] = true;
                                self.dfs_stack.push((
                                    parent_wire,
                                    next_visited_gates,
                                    next_wire_visited as u32,
                                ));
                                wire_found = true;
                                break;
                            }
                        }
                        if wire_found {
                            break;
                        }
                    }
                }
            }

            // Start recursivity of wire by:
            // Pushing the first unvisited neighbour wire of the gates of `wire`.

            for (next_visited_gates, &next_gate_idx) in gates.iter().enumerate() {
                if self.connection_state[s(next_gate_idx)] {
                    self.connection_state[s(next_gate_idx)] = false;
                    let next_gate = &circuit.gates[s(next_gate_idx)];
                    let mut wire_found = false;
                    for (next_wire_visited, &next_wire) in next_gate.wires.iter().enumerate() {
                        if !state[next_wire as usize] {
                            state[next_wire as usize] = true;
                            self.dfs_stack.push((
                                wire,
                                next_visited_gates as u32,
                                next_wire_visited as u32,
                            ));
                            wire_found = true;
                            break;
                        }
                    }
                    if wire_found {
                        break;
                    }
                }
            }
        }
    }
}

/// Implements:
/// - UnionFind
/// - Trivial optimizations
#[derive(Default)]
pub struct CircuitEngineUfT {
    wire_connections: UnionFind,
}

impl CircuitEngine for CircuitEngineUfT {
    fn name(&self) -> &'static str {
        "UnionFind + Trivial"
    }

    fn new_dyn(&self, _: &CircuitImage) -> Box<dyn CircuitEngine> {
        Box::new(Self::default())
    }

    /// Perfoms a single step. Equivalent to `update_wires()` + `update_gates()`
    fn step(&mut self, circuit: &CircuitImage, net_state: &mut CircuitStateNets) {
        self.update_wires(circuit, net_state);
        net_state.update_gates(circuit);
    }
}

impl CircuitEngineUfT {
    /// Given the state of the gates, compute the state of the wires.
    pub fn update_wires(&mut self, circuit: &CircuitImage, state: &mut [bool]) {
        debug_assert!(circuit.net_count() as usize == state.len());
        state[NET_ON as usize] = true;

        self.wire_connections.clear();
        self.wire_connections.extend(circuit.wire_count());

        for buffer_gate in &circuit.buffer_gates {
            let toggled = buffer_gate.controls.iter().all(|&net| state[net as usize]);
            if GateType::Passive.connects_wires(toggled) {
                for &net in &buffer_gate.wires {
                    self.wire_connections.set_parent_unchecked(net, 1);
                }
            }
        }

        for not_gate in &circuit.not_gates {
            let toggled = not_gate.controls.iter().all(|&net| state[net as usize]);
            if GateType::Active.connects_wires(toggled) {
                for &net in &not_gate.wires {
                    self.wire_connections.set_parent_unchecked(net, 1);
                }
            }
        }

        // Connect nets from gates
        for gate in &circuit.non_trivial_gates {
            let toggled = gate.controls.iter().all(|&net| state[net as usize]);
            if gate.ty.connects_wires(toggled) {
                for net in gate.wires.windows(2) {
                    self.wire_connections.alias(net[0], net[1]);
                }
            }
        }

        // Write wire state
        for wire_net in 2..circuit.wire_count() {
            // Since we are iterating in order, and NET_ON is lower than the visited,
            // we can use grandparent instead of root.
            state[wire_net as usize] = self.wire_connections.has_grandparent(wire_net, NET_ON);
        }
    }
}

/// Implements:
/// - Depth First Search
/// - Trivial optimizations
#[derive(Default)]
pub struct CircuitEngineDfs {
    dfs_stack: Vec<u32>,

    /// Weather a certain gate is or not connection its wires.
    connection_state: Vec<bool>,
}

impl CircuitEngine for CircuitEngineDfs {
    fn name(&self) -> &'static str {
        "DFS"
    }

    fn new_dyn(&self, _: &CircuitImage) -> Box<dyn CircuitEngine> {
        Box::new(Self::default())
    }

    /// Perfoms a single step. Equivalent to `update_wires()` + `update_gates()`
    fn step(&mut self, circuit: &CircuitImage, net_state: &mut CircuitStateNets) {
        self.update_wires(circuit, net_state);
        net_state.update_gates(circuit);
    }
}

impl CircuitEngineDfs {
    /// Given the state of the gates, compute the state of the wires.
    pub fn update_wires(&mut self, circuit: &CircuitImage, state: &mut [bool]) {
        debug_assert!(circuit.net_count() as usize == state.len());

        self.connection_state.clear();
        for gate in &circuit.gates {
            let toggled = gate.controls.iter().all(|&net| state[net as usize]);
            self.connection_state.push(gate.ty.connects_wires(toggled));
        }

        // Used as visited map
        state[2..circuit.wire_count() as usize].fill(false);

        self.dfs_stack.clear();
        self.dfs_stack.push(NET_ON);
        state[NET_ON as usize] = true;

        // For all queued nodes (wires)
        while let Some(wire) = self.dfs_stack.pop() {
            // Visit all edges (gates)
            for &gate_idx in &circuit.wires[wire as usize] {
                // If edge is enabled (gate connects wires)
                if self.connection_state[gate_idx as usize] {
                    self.connection_state[gate_idx as usize] = true;
                    // Visit the neighbours
                    for &neighbour in &circuit.gates[gate_idx as usize].wires {
                        if !state[neighbour as usize] {
                            self.dfs_stack.push(neighbour);
                            state[neighbour as usize] = true;
                        }
                    }
                }
            }
        }
    }
}

// /// Circuit state for a `CircuitOptimized`.
// ///
// /// An optimized circuit does not utilize a simple bool array (with the on/off state of the nets)
// /// This is because of optimizations stores the nets in other formats. Specially it groups
// /// on/off states in bitfields to allow the ussage of bitwise operations.
// ///
// /// Even so, fast conversion between this optimized representation and the plain boolean array
// /// is available and efficient (done in a single iteration).
// pub struct CircuitStateOptimized {}
//
// /// Optimized version of circuit for best execution performance.
// ///
// /// Some of the things that the optimizations allows us are
// /// - Remove gates that are permanently on or off
// /// - Use bitwise operations to compute multiple gates at once.
// ///
// /// # Trivial Gate Optimization
// /// A trivial gate can be represented as such:
// /// ```
// /// connect = ctrl0 && ctrl1 && ...;    // Passive Gate
// /// connect = !(ctrl0 && ctrl1 && ...); // Active Gate
// /// if connect { wire0 = 1; wire1 = 1; ... }
// /// ```
// /// which is equivalent to:
// /// ```
// /// connect = ctrl0 && ctrl1 && ...;    // Passive Gate
// /// connect = !(ctrl0 && ctrl1 && ...); // Active Gate
// /// wire0 |= connect;
// /// wire1 |= connect;
// /// ...
// /// ```
// /// and therefore, all the or-assign operations can be applied in bulk with bitwise or operations.
// ///
// /// # Future optimizations to try after benchmarks:
// ///
// /// ## Isolated aliases
// /// All aliases (a = b) where a or b (or both) are unique nets.
// /// A net is unique if it's only aliased by a single gate (ignoring trivial not & buffer gates).
// ///
// /// This allows to do:
// /// ```
// /// for (isolated_a, b) in isolated_aliases:
// ///   if (connected) b = isolated_a;
// /// apply(alias_groups);
// /// for (isolated_a, b) in isolated_aliases:
// ///   if (connected) isolated_a = b;
// /// ```
// /// Reducing this way the amout and size of the alias groups.
// ///
// pub struct CircuitOptimized {
//     /// Dense mapping to convert from `CircuitStateOptimized` to `CircuitState`
//     from_optimized: Vec<u32>,
//
//     /// Dense mapping to convert from `CircuitState` to `CircuitStateOptimized`
//     to_optimized: Box<[u32]>,
//
//     passive_gate_controls: Vec<SmallVec<[(u32, MASK); 2]>>,
//     active_gate_controls: Vec<SmallVec<[(u32, MASK); 2]>>,
//
//     // for w in wires:
//     //   for g in w.gates:
//     //     (g_control_id, g.other_wires)
//     w: u32,
//     // /// A not gate (an active gate with a NET_ON connected as a wire)
//     // not_gates: Vec<BitfieldGate>,
//     //
//     // /// A buffer gate (an active gate with a NET_ON connected as a wire)
//     // buffer_gates: Vec<BitfieldGate>,
//     //
//     // /// Contains the remaining aliases to be done after the applying the `trivial_aliases`.
//     // ///
//     // /// Aliases are grouped by their corresponding "connected component" (graph theory).
//     // /// This allows us to work with independent smaller aliases graphs instead of a single huge one.
//     // aliases_groups: Vec<Vec<u32>>,
// }
//
// // Gate representation for `CircuitOptimized`
// struct BitfieldGate {
//     controls: SmallVec<[(u32, MASK); 2]>,
//     wire: SmallVec<[(u32, MASK); 2]>,
// }
//
// struct GrateGroup {
//     controls: Vec<(u32, MASK)>,
//     // (wire_net, )
//     wires: Vec<(u32, u32)>,
// }
//
// impl CircuitOptimized {
//     pub fn new(circuit: &Circuit) -> CircuitOptimized {
//         let from_optimized = Vec::new();
//         let to_optimized = vec![0; circuit.net_count() as usize].into_boxed_slice();
//         let trivial_aliases = Vec::new();
//         let aliases_groups = Vec::new();
//
//         CircuitOptimized {
//             from_optimized,
//             to_optimized,
//             trivial_aliases,
//             aliases_groups,
//         }
//     }
// }
