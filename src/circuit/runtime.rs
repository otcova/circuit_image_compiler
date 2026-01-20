use super::*;

/// Executes a circuit directly without applying any previous preprocessing (or optimizations).
///
/// The cost of performing the circuit optimizations needed for "CircuitInterpreter"
/// are very small and can be ignored. So use `CircuitInterpreter` instead in case of wanting
/// a fast simulation.
///
/// The main purpose of this `UnoptimizedCircuitInterpreter` is to be used as a:
/// - Didactic Example: Its a staight forward implementation, working directly with the nets and gates.
/// - Base Comparison for more complex executors.
#[derive(Default)]
pub struct UnoptimizedCircuitInterpreter {
    // TODO: Use a dfs instead to only iterate the nets that have been connected to NET_ON.
    wire_connections: UnionFind,
}

impl UnoptimizedCircuitInterpreter {
    /// Perfoms a single step. Equivalent to `update_wires()` + `update_gates()`
    pub fn step(&mut self, circuit: &Circuit, net_state: &mut CircuitState) {
        self.update_wires(circuit, net_state);
        self.update_gates(circuit, net_state);
    }

    /// Given the state of the gates, compute the state of the wires.
    pub fn update_wires(&mut self, circuit: &Circuit, net_state: &mut CircuitState) {
        debug_assert!(circuit.net_count() as usize == net_state.len());
        net_state[NET_ON as usize] = true;

        self.wire_connections.clear();
        self.wire_connections.extend(circuit.net_count()); // TODO: Why wire_count does not work

        // Connect nets from gates
        for gate in &circuit.gates {
            let toggled = gate.controls.iter().all(|&net| net_state[net as usize]);
            match (gate.ty, toggled) {
                (GateType::Active, false) | (GateType::Passive, true) => {
                    for net in gate.wires.windows(2) {
                        self.wire_connections.alias(net[0], net[1]);
                    }
                }
                _ => {}
            }
        }

        // Write wire state
        for wire_net in 2..circuit.wire_count() {
            net_state[wire_net as usize] = self.wire_connections.root(wire_net) == NET_ON;
        }
    }

    /// Given the state of the wires, compute the state of the gates.
    pub fn update_gates(&mut self, circuit: &Circuit, net_state: &mut CircuitState) {
        debug_assert!(circuit.net_count() as usize == net_state.len());
        for (gate_i, gate) in circuit.gates.iter().enumerate() {
            let gate_net = gate_i + circuit.wire_count() as usize;
            net_state[gate_net] = gate.wires.iter().any(|&net| net_state[net as usize]);
        }
    }
}

/// Circuit state for a `CircuitOptimized`.
///
/// An optimized circuit does not utilize a simple bool array (with the on/off state of the nets)
/// This is because of optimizations stores the nets in other formats. Specially it groups
/// on/off states in bitsets to allow the ussage of bitwise operations.
///
/// Even so, fast conversion between this optimized representation and the plain boolean array
/// is available and efficient (done in a single iteration).
pub struct CircuitStateOptimized {}

/// Optimized version of circuit for best execution performance.
///
/// Some of the things that the optimizations allows us are
/// - Remove gates that are permanently on or off
/// - Use bitwise operations to compute multiple gates at once.
///
/// # Future optimizations to try after benchmarks:
///
/// ## Isolated aliases
/// All aliases (a = b) where a or b (or both) are unique nets.
/// A net is unique if it's only aliased by a single gate (ignoring trivial not & buffer gates).
///
/// This allows to do:
/// ```
/// for (isolated_a, b) in isolated_aliases:
///   if (connected) b = isolated_a;
/// apply(alias_groups);
/// for (isolated_a, b) in isolated_aliases:
///   if (connected) isolated_a = b;
/// ```
/// Reducing this way the amout and size of the alias groups.
///
pub struct CircuitOptimized {
    /// Dense mapping to convert from `CircuitStateOptimized` to `CircuitState`
    from_optimized: Vec<u32>,

    /// Dense mapping to convert from `CircuitState` to `CircuitStateOptimized`
    to_optimized: Box<[u32]>,

    /// All aliases generated from not or buffer gates.
    /// A not gate is an active gate with a NET_ON connected as a wire.
    /// A buffer gate is a passive gate with a NET_ON connected as a wire.
    ///
    /// # Optimization Description
    /// A trivial gate can be represented as such:
    /// ```
    /// connect = ctrl0 && ctrl1 && ...;    // Passive Gate
    /// connect = !(ctrl0 && ctrl1 && ...); // Active Gate
    /// if connect { wire0 = 1; wire1 = 1; ... }
    /// ```
    /// which is equivalent to:
    /// ```
    /// connect = ctrl0 && ctrl1 && ...;    // Passive Gate
    /// connect = !(ctrl0 && ctrl1 && ...); // Active Gate
    /// wire0 |= connect;
    /// wire1 |= connect;
    /// ...
    /// ```
    /// and therefore, all the or-assign operations can be applied in bulk with bitwise or operations.
    trivial_aliases: Vec<u32>,

    /// Contains the remaining aliases to be done after the applying the `trivial_aliases`.
    ///
    /// Aliases are grouped by their corresponding "connected component" (graph theory).
    /// This allows us to work with independent smaller aliases graphs instead of a single huge one.
    aliases_groups: Vec<Vec<u32>>,
}

impl CircuitOptimized {
    pub fn new(circuit: &Circuit) -> CircuitOptimized {
        let from_optimized = Vec::new();
        let to_optimized = vec![0; circuit.net_count() as usize].into_boxed_slice();
        let trivial_aliases = Vec::new();
        let aliases_groups = Vec::new();

        CircuitOptimized {
            from_optimized,
            to_optimized,
            trivial_aliases,
            aliases_groups,
        }
    }
}
