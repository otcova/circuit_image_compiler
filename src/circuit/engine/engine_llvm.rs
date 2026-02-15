use super::*;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::module::Module;
use inkwell::values::IntValue;
use inkwell::{AddressSpace, OptimizationLevel};

/// Implements:
/// - Depth First Search
#[derive(Clone)]
pub struct CircuitEngineLlvm {
    // We need to keep context alive since we have a reference to it (jit_step)
    _context: Arc<(ExecutionEngine<'static>, Context)>,
    jit_step: StepFunc,

    dfs_stack: Vec<u32>,

    /// Weather a certain gate is or not connection its wires.
    non_trivial_is_connected: Vec<bool>,
}

unsafe impl Send for CircuitEngineLlvm {}

impl CircuitEngine for CircuitEngineLlvm {
    fn name(&self) -> &'static str {
        "DFS LLVM"
    }

    fn clone_dyn(&self) -> Box<dyn CircuitEngine> {
        Box::new(self.clone())
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

impl CircuitEngineLlvm {
    pub fn new(circuit: &CircuitImage) -> Self {
        let context = Context::create();
        let module = context.create_module("");
        let execution_engine = module
            .create_jit_execution_engine(OptimizationLevel::Aggressive)
            .unwrap();

        let codegen = CodeGen {
            context: &context,
            module,
            builder: context.create_builder(),
            execution_engine: execution_engine.clone(),
        };

        let jit_step = {
            let jit_step = codegen
                .jit_compile_step(circuit)
                .expect("Unable to JIT compile");
            unsafe { jit_step.as_raw() }
        };

        let exec_engine = unsafe {
            std::mem::transmute::<ExecutionEngine<'_>, ExecutionEngine<'static>>(execution_engine)
        };

        drop(codegen);

        Self {
            #[allow(clippy::arc_with_non_send_sync)]
            _context: Arc::new((exec_engine, context)),
            jit_step,
            dfs_stack: Default::default(),
            non_trivial_is_connected: vec![false; circuit.non_trivial_gates.len()],
        }
    }

    /// Given the state of the gates, compute the state of the wires.
    pub fn update_wires(&mut self, state: &mut CircuitState) {
        // Done by jit_compile_step
        // state.apply_inputs();

        let nets = state.nets.as_concat_mut();

        unsafe {
            (self.jit_step)(
                nets.as_mut_ptr() as *mut u8,
                self.non_trivial_is_connected.as_mut_ptr() as *mut u8,
            );
        }

        // Done by jit_step
        // for buffer_gate in &state.image.buffer_gates {
        //     let toggled = buffer_gate.controls.iter().all(|&net| nets[net as usize]);
        //     if GateType::Passive.connects_wires(toggled) {
        //         for &net in &buffer_gate.wires {
        //             nets[net as usize] = true;
        //         }
        //     }
        // }
        // for not_gate in &state.image.not_gates {
        //     let toggled = not_gate.controls.iter().all(|&net| nets[net as usize]);
        //     if GateType::Active.connects_wires(toggled) {
        //         for &net in &not_gate.wires {
        //             nets[net as usize] = true;
        //         }
        //     }
        // }
        // for (idx, gate) in state.image.non_trivial_gates.iter().enumerate() {
        //     let is_toggled = gate.controls.iter().all(|&net| nets[net as usize]);
        //     let is_connected = gate.ty.connects_wires(is_toggled);
        //     self.non_trivial_is_connected[idx] = is_connected;
        // }

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

/// Convenience type alias for the `sum` function.
///
/// Calling this is innately `unsafe` because there's no guarantee it doesn't
/// do `unsafe` operations internally.
type StepFunc = unsafe extern "C" fn(*mut u8, *mut u8);

struct CodeGen<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    execution_engine: ExecutionEngine<'ctx>,
}

impl<'ctx> CodeGen<'ctx> {
    fn jit_compile_step(&self, circuit: &CircuitImage) -> Option<JitFunction<'_, StepFunc>> {
        let ptr_type = self.context.ptr_type(AddressSpace::default());
        let i8_type = self.context.i8_type();
        let i32_type = self.context.i32_type();
        let void_type = self.context.void_type();

        // let const_false = i8_type.const_zero();
        let const_true = i8_type.const_int(1, false);

        let fn_type = void_type.fn_type(&[ptr_type.into(), ptr_type.into()], false);
        let function = self.module.add_function("step", fn_type, None);
        let basic_block = self.context.append_basic_block(function, "entry");

        self.builder.position_at_end(basic_block);

        let nets = function.get_nth_param(0)?.into_pointer_value();
        let connected = function.get_nth_param(1)?.into_pointer_value();

        let set_connected = |idx: u32, value: IntValue<'ctx>| {
            let offset = i32_type.const_int(idx as u64, false);
            let ptr = unsafe { connected.const_in_bounds_gep(i8_type, &[offset]) };
            self.builder.build_store(ptr, value).unwrap();
        };

        let get_net_ptr = |net: u32| {
            let offset = i32_type.const_int(net as u64, false);
            unsafe { nets.const_in_bounds_gep(i8_type, &[offset]) }
        };

        let get_net = |net: u32| {
            self.builder
                .build_load(i8_type, get_net_ptr(net), "")
                .unwrap()
                .into_int_value()
        };

        let set_net = |net: u32, value: IntValue<'ctx>| {
            self.builder.build_store(get_net_ptr(net), value).unwrap();
        };

        // Apply inputs
        for net_id in 0..circuit.wire_count() {
            let input_id = circuit.net_count() + net_id;
            set_net(net_id, get_net(input_id));
        }

        // Compute buffer gates
        for buffer_gate in &circuit.not_gates {
            let mut toggled = const_true;

            for &control in &buffer_gate.controls {
                let val = get_net(control);
                toggled = self.builder.build_and(toggled, val, "").unwrap();
            }

            for &net in &buffer_gate.wires {
                let val = self.builder.build_or(get_net(net), toggled, "").unwrap();
                set_net(net, val);
            }
        }

        // Compute not gates
        for not_gate in &circuit.buffer_gates {
            let mut toggled = const_true;

            for &control in &not_gate.controls {
                let val = get_net(control);
                toggled = self.builder.build_and(toggled, val, "").unwrap();
            }

            let not_toggled = self.builder.build_xor(toggled, const_true, "").unwrap();

            for &net in &not_gate.wires {
                let val = self
                    .builder
                    .build_or(get_net(net), not_toggled, "")
                    .unwrap();
                set_net(net, val);
            }
        }

        // Compute the non trivial gates connected states
        for (idx, gate) in circuit.non_trivial_gates.iter().enumerate() {
            let mut is_toggled = const_true;

            for &control in &gate.controls {
                let val = get_net(control);
                is_toggled = self.builder.build_and(is_toggled, val, "").unwrap();
            }

            let is_connected = if gate.ty == GateType::Active {
                is_toggled
            } else {
                self.builder.build_xor(is_toggled, const_true, "").unwrap()
            };

            set_connected(idx as u32, is_connected);
        }

        self.builder.build_return(None).unwrap();

        unsafe { self.execution_engine.get_function("step").ok() }
    }
}
