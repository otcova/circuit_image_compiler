use eframe::{
    egui::{self, Vec2},
    egui_glow::check_for_gl_error,
    glow::{self, HasContext},
};

pub use camera::*;

use crate::circuit::{CircuitImage, CircuitState};

mod camera;

pub struct CircuitCanvas {
    tex_size: Vec2,
    tex_image: glow::Texture,
    tex_nets: glow::Texture,
    tex_net_state: glow::Texture,
    buffer_net_state: glow::Buffer,

    shader_program: glow::Program,
    vertex_array: glow::VertexArray,

    pub camera: Camera,
    pub selected_net: u32,

    /// Temporary buffer to compute if a gate is or not connected.
    gate_state: Vec<bool>,
}

impl CircuitCanvas {
    pub fn new(gl: &glow::Context) -> Self {
        check_for_gl_error!(gl);

        let (shader_program, vertex_array) = init_shaders(gl);

        let tex_image = unsafe {
            let texture = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));

            use glow::*;
            gl.tex_parameter_i32(TEXTURE_2D, TEXTURE_MIN_FILTER, LINEAR_MIPMAP_NEAREST as i32);
            gl.tex_parameter_i32(TEXTURE_2D, TEXTURE_MAG_FILTER, NEAREST as i32);

            let background = [0., 0., 0., 255.];
            gl.tex_parameter_i32(TEXTURE_2D, TEXTURE_WRAP_S, CLAMP_TO_BORDER as i32);
            gl.tex_parameter_i32(TEXTURE_2D, TEXTURE_WRAP_T, CLAMP_TO_BORDER as i32);
            gl.tex_parameter_f32_slice(TEXTURE_2D, TEXTURE_BORDER_COLOR, &background);

            // Configure RGB format without padding. (align at 1 byte)
            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
            check_for_gl_error!(gl);
            texture
        };

        let tex_nets = unsafe {
            let texture = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));

            use glow::*;
            gl.tex_parameter_i32(TEXTURE_2D, TEXTURE_MIN_FILTER, NEAREST as i32);
            gl.tex_parameter_i32(TEXTURE_2D, TEXTURE_MAG_FILTER, NEAREST as i32);
            gl.tex_parameter_i32(TEXTURE_2D, TEXTURE_WRAP_S, CLAMP_TO_EDGE as i32);
            check_for_gl_error!(gl);
            texture
        };

        let buffer_net_state = unsafe { gl.create_buffer().unwrap() };

        let tex_net_state = unsafe {
            let texture = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_BUFFER, Some(texture));
            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
            check_for_gl_error!(gl);
            texture
        };

        CircuitCanvas {
            tex_size: Vec2::ZERO,
            tex_image,
            tex_nets,
            tex_net_state,
            buffer_net_state,
            shader_program,
            vertex_array,
            camera: Camera::new(),
            selected_net: 0,
            gate_state: Vec::new(),
        }
    }

    /// Deltes internal gl objects
    ///
    /// # Safety
    /// Should be the last method executed before drop
    pub fn delete(&self, gl: &glow::Context) {
        unsafe {
            gl.delete_texture(self.tex_image);
            gl.delete_texture(self.tex_nets);
            gl.delete_texture(self.tex_net_state);
            gl.delete_buffer(self.buffer_net_state);

            gl.delete_vertex_array(self.vertex_array);
            gl.delete_program(self.shader_program);
            check_for_gl_error!(gl);
        }
    }

    // pub fn load_circuit(&mut self, gl: &glow::Context, circuit: &CircuitImage, camera: Camera) {
    //     self.camera = camera;
    pub fn load_circuit(&mut self, gl: &glow::Context, circuit: &CircuitImage, camera: Camera) {
        self.tex_size = Vec2::new(circuit.width() as f32, circuit.height() as f32);
        self.camera = camera;

        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(self.tex_image));
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGB as i32,
                circuit.width() as i32,
                circuit.height() as i32,
                0,
                glow::RGB,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(circuit.colors().as_raw())),
            );
            gl.generate_mipmap(glow::TEXTURE_2D);

            gl.bind_texture(glow::TEXTURE_2D, Some(self.tex_nets));
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::R32UI as i32,
                circuit.width() as i32,
                circuit.height() as i32,
                0,
                glow::RED_INTEGER,
                glow::UNSIGNED_INT,
                glow::PixelUnpackData::Slice(Some(bytemuck::cast_slice(circuit.nets()))),
            );

            gl.bind_buffer(glow::TEXTURE_BUFFER, Some(self.buffer_net_state));
            gl.buffer_data_size(
                glow::TEXTURE_BUFFER,
                (2 * circuit.net_count() + circuit.gate_count()) as i32,
                glow::DYNAMIC_DRAW,
            );

            gl.bind_texture(glow::TEXTURE_BUFFER, Some(self.tex_net_state));
            gl.tex_buffer(glow::TEXTURE_BUFFER, glow::R8, Some(self.buffer_net_state));

            gl.use_program(Some(self.shader_program));
            gl.uniform_3_f32(
                Some(
                    &gl.get_uniform_location(self.shader_program, "power_color")
                        .unwrap(),
                ),
                circuit.power_color()[0] as f32 / 255.,
                circuit.power_color()[1] as f32 / 255.,
                circuit.power_color()[2] as f32 / 255.,
            );
            gl.uniform_3_f32(
                Some(
                    &gl.get_uniform_location(self.shader_program, "active_color")
                        .unwrap(),
                ),
                circuit.active_gate_color()[0] as f32 / 255.,
                circuit.active_gate_color()[1] as f32 / 255.,
                circuit.active_gate_color()[2] as f32 / 255.,
            );
            gl.uniform_3_f32(
                Some(
                    &gl.get_uniform_location(self.shader_program, "passive_color")
                        .unwrap(),
                ),
                circuit.passive_gate_color()[0] as f32 / 255.,
                circuit.passive_gate_color()[1] as f32 / 255.,
                circuit.passive_gate_color()[2] as f32 / 255.,
            );
            gl.uniform_1_u32(
                Some(
                    &gl.get_uniform_location(self.shader_program, "net_count")
                        .unwrap(),
                ),
                circuit.net_count(),
            );
            gl.uniform_1_u32(
                Some(
                    &gl.get_uniform_location(self.shader_program, "wire_count")
                        .unwrap(),
                ),
                circuit.wire_count(),
            );

            check_for_gl_error!(gl);
        }
    }

    pub fn load_circuit_state(&mut self, gl: &glow::Context, state: &CircuitState) {
        let nets = state.nets.get();
        let nets_and_inputs = state.nets.as_concat();

        unsafe {
            gl.bind_buffer(glow::TEXTURE_BUFFER, Some(self.buffer_net_state));
            gl.buffer_sub_data_u8_slice(
                glow::TEXTURE_BUFFER,
                0,
                bytemuck::cast_slice(nets_and_inputs),
            );

            // TODO: remove this
            // gl.buffer_data_u8_slice(
            //     glow::TEXTURE_BUFFER,
            //     bytemuck::cast_slice(state.nets.as_concat()),
            //     glow::DYNAMIC_DRAW,
            // );

            self.gate_state.clear();
            for gate in &state.image.gates {
                let is_toggled = gate.controls.iter().all(|&net| nets[net as usize]);
                self.gate_state.push(is_toggled);
            }

            gl.buffer_sub_data_u8_slice(
                glow::TEXTURE_BUFFER,
                nets_and_inputs.len() as i32,
                bytemuck::cast_slice(&self.gate_state),
            );
            check_for_gl_error!(gl);
        }
    }

    pub fn render_callback(&self, surface_rect: egui::Rect) -> impl Fn(&glow::Context) + 'static {
        let surface_size = surface_rect.size();

        let uv_min = self.camera.surface_to_texel(Vec2::ZERO, surface_size) / self.tex_size;
        let uv_max = self.camera.surface_to_texel(surface_size, surface_size) / self.tex_size;
        let uv_size = uv_max - uv_min;

        // Clone locals so we can move them into the paint callback:
        let program = self.shader_program;
        let vertex_array = self.vertex_array;
        let tex_image = self.tex_image;
        let tex_nets = self.tex_nets;
        let tex_net_state = self.tex_net_state;
        let selected_net = self.selected_net;
        let texel_size = self.camera.surface_pixels_per_texel();

        move |gl| unsafe {
            gl.use_program(Some(program));

            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(tex_image));
            gl.uniform_1_i32(
                Some(&gl.get_uniform_location(program, "tex_circuit").unwrap()),
                0,
            );
            gl.active_texture(glow::TEXTURE1);
            gl.bind_texture(glow::TEXTURE_2D, Some(tex_nets));
            gl.uniform_1_i32(
                Some(&gl.get_uniform_location(program, "tex_nets").unwrap()),
                1,
            );
            gl.active_texture(glow::TEXTURE2);
            gl.bind_texture(glow::TEXTURE_BUFFER, Some(tex_net_state));
            gl.uniform_1_i32(
                Some(&gl.get_uniform_location(program, "tex_net_state").unwrap()),
                2,
            );
            gl.uniform_2_f32(
                Some(&gl.get_uniform_location(program, "uv_min").unwrap()),
                uv_min.x,
                uv_min.y,
            );
            gl.uniform_2_f32(
                Some(&gl.get_uniform_location(program, "uv_size").unwrap()),
                uv_size.x,
                uv_size.y,
            );

            gl.uniform_1_f32(
                Some(&gl.get_uniform_location(program, "texel_size").unwrap()),
                texel_size,
            );

            let pixel_size = uv_size / surface_size;
            gl.uniform_2_f32(
                Some(&gl.get_uniform_location(program, "pixel_size").unwrap()),
                pixel_size.x,
                pixel_size.y,
            );
            gl.uniform_1_u32(
                Some(&gl.get_uniform_location(program, "selected_net").unwrap()),
                selected_net,
            );

            gl.bind_vertex_array(Some(vertex_array));
            gl.draw_arrays(glow::TRIANGLES, 0, 3);
            check_for_gl_error!(gl);

            gl.bind_buffer(glow::TEXTURE_BUFFER, None);
            gl.bind_texture(glow::TEXTURE_BUFFER, None);
            gl.bind_texture(glow::TEXTURE_2D, None);
            gl.use_program(None);
        }
    }
}

fn init_shaders(gl: &glow::Context) -> (glow::Program, glow::VertexArray) {
    let shader_version = if cfg!(target_arch = "wasm32") {
        "#version 300 es"
    } else {
        "#version 330"
    };

    unsafe {
        let program = gl.create_program().expect("Cannot create program");

        let shader_sources = [
            (
                "Vertex Shader",
                glow::VERTEX_SHADER,
                include_str!("shader.vert"),
            ),
            (
                "Fragment Shader",
                glow::FRAGMENT_SHADER,
                include_str!("shader.frag"),
            ),
        ];

        let shaders: Vec<_> = shader_sources
            .iter()
            .map(|(shader_name, shader_type, shader_source)| {
                let shader = gl
                    .create_shader(*shader_type)
                    .expect("Cannot create shader");
                gl.shader_source(shader, &format!("{shader_version}\n{shader_source}"));
                gl.compile_shader(shader);
                assert!(
                    gl.get_shader_compile_status(shader),
                    "Failed to compile {shader_name}: {}",
                    gl.get_shader_info_log(shader)
                );
                gl.attach_shader(program, shader);
                shader
            })
            .collect();

        gl.link_program(program);
        assert!(
            gl.get_program_link_status(program),
            "{}",
            gl.get_program_info_log(program)
        );

        for shader in shaders {
            gl.detach_shader(program, shader);
            gl.delete_shader(shader);
        }

        let vertex_array = gl
            .create_vertex_array()
            .expect("Cannot create vertex array");

        check_for_gl_error!(gl);
        (program, vertex_array)
    }
}
