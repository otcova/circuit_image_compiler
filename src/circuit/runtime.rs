use crate::bench::bench_seconds;

use super::*;
use std::{
    ops::{Deref, DerefMut},
    time::Duration,
};

/// Responisble of executing a circuit.
pub trait CircuitRuntime {
    fn name(&self) -> &'static str;

    /// Crates a new runtime with the configuration of this one for the given circuit.
    fn new_dyn(&self, circuit: &CircuitImage) -> Box<dyn CircuitRuntime>;

    /// Use step_n for faster iteration
    fn step(&mut self, circuit: &CircuitImage, state: &mut CircuitStateNets);

    /// This is faster than step_n (if n > 1) since runtimes may:
    /// - Apply optimizations that allow for jumping multiple steps at once.
    /// - Need to transform the state into
    fn step_n(&mut self, circuit: &CircuitImage, state: &mut CircuitStateNets, n: u64) {
        for _ in 0..n {
            self.step(circuit, state);
        }
    }

    /// Runs a steps for the amount of time specified.
    /// Returns the steps/second.
    fn bench(&mut self, circuit: &CircuitImage, time: Duration) -> f32 {
        let step_by_n: u64 = 32;
        let state = &mut CircuitStateNets::new(circuit);
        step_by_n as f32 / bench_seconds(|| self.step_n(circuit, state, step_by_n), time)
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

#[derive(Default)]
pub struct CircuitRuntimeUnionFind {
    wire_connections: UnionFind,
}

impl CircuitRuntime for CircuitRuntimeUnionFind {
    fn name(&self) -> &'static str {
        "UnionFind"
    }

    fn new_dyn(&self, _: &CircuitImage) -> Box<dyn CircuitRuntime> {
        Box::new(Self::default())
    }

    /// Perfoms a single step. Equivalent to `update_wires()` + `update_gates()`
    fn step(&mut self, circuit: &CircuitImage, net_state: &mut CircuitStateNets) {
        self.update_wires(circuit, net_state);
        net_state.update_gates(circuit);
    }
}

impl CircuitRuntimeUnionFind {
    /// Given the state of the gates, compute the state of the wires.
    pub fn update_wires(&mut self, circuit: &CircuitImage, net_state: &mut [bool]) {
        debug_assert!(circuit.net_count() as usize == net_state.len());
        net_state[NET_ON as usize] = true;

        self.wire_connections.clear();
        self.wire_connections.extend(circuit.wire_count());

        // Connect nets from gates
        for gate in &circuit.gates {
            let toggled = gate.controls.iter().all(|&net| net_state[net as usize]);
            if gate.ty.connects_wires(toggled) {
                for net in gate.wires.windows(2) {
                    self.wire_connections.alias(net[0], net[1]);
                }
            }
        }

        // Write wire state
        for wire_net in 2..circuit.wire_count() {
            net_state[wire_net as usize] = self.wire_connections.root(wire_net) == NET_ON;
        }
    }
}

#[derive(Default)]
pub struct CircuitRuntimeDfs {
    dfs_stack: Vec<u32>,

    /// Weather a certain gate is or not connection its wires.
    connection_state: Vec<bool>,
}

impl CircuitRuntime for CircuitRuntimeDfs {
    fn name(&self) -> &'static str {
        "DFS"
    }

    fn new_dyn(&self, _: &CircuitImage) -> Box<dyn CircuitRuntime> {
        Box::new(Self::default())
    }

    /// Perfoms a single step. Equivalent to `update_wires()` + `update_gates()`
    fn step(&mut self, circuit: &CircuitImage, net_state: &mut CircuitStateNets) {
        self.update_wires(circuit, net_state);
        net_state.update_gates(circuit);
    }
}

impl CircuitRuntimeDfs {
    /// Given the state of the gates, compute the state of the wires.
    pub fn update_wires(&mut self, circuit: &CircuitImage, net_state: &mut [bool]) {
        debug_assert!(circuit.net_count() as usize == net_state.len());

        self.connection_state.clear();
        for gate in &circuit.gates {
            let toggled = gate.controls.iter().all(|&net| net_state[net as usize]);
            self.connection_state.push(gate.ty.connects_wires(toggled));
        }

        // Used as visited map
        net_state[..circuit.wire_count() as usize].fill(false);

        self.dfs_stack.clear();
        self.dfs_stack.push(NET_ON);
        net_state[NET_ON as usize] = true;

        // For all queued nodes (wires)
        while let Some(wire) = self.dfs_stack.pop() {
            // Visit all edges (gates)
            for &gate_idx in &circuit.wires[wire as usize] {
                // If edge is enabled (gate connects wires)
                if self.connection_state[gate_idx as usize] {
                    // Visit the neighbours
                    for &neighbour in &circuit.gates[gate_idx as usize].wires {
                        if !net_state[neighbour as usize] {
                            self.dfs_stack.push(neighbour);
                            net_state[neighbour as usize] = true;
                        }
                    }
                }
            }
        }

        // self.wire_connections.clear();
        // self.wire_connections.extend(circuit.wire_count());
        //
        // // Connect nets from gates
        // for gate in &circuit.gates {
        //     let toggled = gate.controls.iter().all(|&net| net_state[net as usize]);
        //     match (gate.ty, toggled) {
        //         (GateType::Active, false) | (GateType::Passive, true) => {
        //             for net in gate.wires.windows(2) {
        //                 self.wire_connections.alias(net[0], net[1]);
        //             }
        //         }
        //         _ => {}
        //     }
        // }
        //
        // // Write wire state
        // for wire_net in 2..circuit.wire_count() {
        //     net_state[wire_net as usize] = self.wire_connections.root(wire_net) == NET_ON;
        // }
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
