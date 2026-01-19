use std::sync::Arc;

use eframe::{
    egui::{self, Key, KeyboardShortcut, Modifiers, PaintCallback, PointerButton, Sense, Ui, Vec2},
    egui_glow,
    glow::{self, HasContext},
};
use image::{EncodableLayout, ImageReader};

mod circuit;
use circuit::*;

mod camera;
use camera::*;

fn main() {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([800.0, 600.0]),
        renderer: eframe::Renderer::Glow,
        ..Default::default()
    };
    eframe::run_native(
        "Painting comes alive",
        options,
        Box::new(|cc| Ok(Box::new(MyEguiApp::new(cc)))),
    )
    .unwrap();
}

struct MyEguiApp {
    tex_image: glow::Texture,
    tex_nets: glow::Texture,
    tex_state: glow::Texture,
    buffer_state: glow::Buffer,

    circuit: Circuit,
    circuit_state: Vec<bool>,
    interpreter: CircuitInterpreter,

    cursor: Option<(u32, u32)>,

    program: glow::Program,
    vertex_array: glow::VertexArray,

    camera: Camera,
}

impl MyEguiApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let gl = cc.gl.as_ref().expect("Glow backend is needed");

        let img = ImageReader::open("circuits/test.png")
            // let img = ImageReader::open("circuits/playground.png")
            .unwrap()
            .decode()
            .unwrap()
            .to_rgb8();

        let circuit = Circuit::new(img);

        let interpreter = CircuitInterpreter::default();
        let mut circuit_state = Vec::new();
        interpreter.initial_state(&circuit, &mut circuit_state);

        let tex_image = unsafe {
            let texture = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));

            use glow::*;
            gl.tex_parameter_i32(TEXTURE_2D, TEXTURE_MIN_FILTER, NEAREST as i32);
            gl.tex_parameter_i32(TEXTURE_2D, TEXTURE_MAG_FILTER, NEAREST as i32);

            let background = [0., 0., 0., 255.];
            gl.tex_parameter_i32(TEXTURE_2D, TEXTURE_WRAP_S, CLAMP_TO_BORDER as i32);
            gl.tex_parameter_i32(TEXTURE_2D, TEXTURE_WRAP_T, CLAMP_TO_BORDER as i32);
            gl.tex_parameter_f32_slice(TEXTURE_2D, TEXTURE_BORDER_COLOR, &background);

            // Configure RGB format without padding. (align at 1 byte)
            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);

            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGB as i32,
                circuit.image.width() as i32,
                circuit.image.height() as i32,
                0,
                glow::RGB,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(circuit.image.colors().as_bytes())),
            );
            texture
        };

        let tex_nets = unsafe {
            let texture = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));

            use glow::*;
            gl.tex_parameter_i32(TEXTURE_2D, TEXTURE_MIN_FILTER, NEAREST as i32);
            gl.tex_parameter_i32(TEXTURE_2D, TEXTURE_MAG_FILTER, NEAREST as i32);
            gl.tex_parameter_i32(TEXTURE_1D, TEXTURE_WRAP_S, CLAMP_TO_EDGE as i32);

            let raw_nets = bytemuck::cast_slice(circuit.image.nets());
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::R32UI as i32,
                circuit.image.width() as i32,
                circuit.image.height() as i32,
                0,
                glow::RED_INTEGER,
                glow::UNSIGNED_INT,
                glow::PixelUnpackData::Slice(Some(raw_nets)),
            );
            texture
        };

        let buffer_state = unsafe {
            let buffer = gl.create_buffer().unwrap();
            gl.bind_buffer(glow::TEXTURE_BUFFER, Some(buffer));
            gl.buffer_data_u8_slice(
                glow::TEXTURE_BUFFER,
                bytemuck::cast_slice(&circuit_state),
                glow::DYNAMIC_DRAW,
            );
            buffer
        };

        let tex_state = unsafe {
            let texture = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_BUFFER, Some(texture));
            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
            gl.tex_buffer(glow::TEXTURE_BUFFER, glow::R8UI, Some(buffer_state));
            texture
        };

        let (program, vertex_array) = Self::init_shadersnew(gl);

        let mut camera = Camera::new();
        let tex_size = Vec2::new(circuit.image.width() as f32, circuit.image.height() as f32);
        camera.position = tex_size / 2.;
        camera.set_surface_pixels_per_texel(500. / tex_size.y);

        Self {
            tex_image,
            tex_nets,
            tex_state,
            buffer_state,
            circuit_state,
            circuit,
            interpreter,
            cursor: None,
            program,
            vertex_array,
            camera,
        }
    }
}

impl eframe::App for MyEguiApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        let gl = frame.gl().unwrap();

        if ctx.input(|i| i.key_pressed(Key::Escape)) {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }

        egui::SidePanel::left("left_bar")
            .min_width(230.0)
            .resizable(false)
            .show(ctx, |ui| {
                ui.heading("Tools?");
                ui.separator();

                egui::ScrollArea::vertical()
                    .auto_shrink([false; 2])
                    .show(ui, |ui| self.left_panel(ctx, ui, gl));
            });

        egui::CentralPanel::default()
            .frame(egui::Frame::new())
            .show(ctx, |ui| {
                self.custom_painting(ui);
            });
    }

    fn on_exit(&mut self, gl: Option<&glow::Context>) {
        if let Some(gl) = gl {
            unsafe {
                gl.delete_program(self.program);
                gl.delete_vertex_array(self.vertex_array);
            }
        }
    }
}

impl MyEguiApp {
    fn left_panel(&mut self, ctx: &egui::Context, ui: &mut Ui, gl: &glow::Context) {
        if let Some((x, y)) = self.cursor {
            ui.strong(format!("pos:   {x}, {y}"));

            let pixel = self.circuit.image.pixel(x, y);
            if let Some(net) = pixel.net() {
                ui.strong(format!("net: {:?}", net));

                if let Some(gate) = self.circuit.get_gate(net) {
                    ui.strong(format!("gate type: {:?}", gate.ty));
                    ui.strong(format!("gate inputs: {:?}", gate.inputs));
                    ui.strong(format!("gate outputs: {:?}", gate.outputs));
                }
            }

            if let Some(&color) = self.circuit.image.colors().get_pixel_checked(x, y) {
                ui.strong(format!("color: {:?}", color));
                ui.strong(format!("saturation: {:.0}%", 100. * hsv_saturation(color)));
                ui.strong(format!("value: {:.0}%", 100. * hsv_value(color)));
            }

            ui.separator();
        }

        ui.strong(format!("net count: {:?}", self.circuit.net_count()));

        ui.separator();

        let shortcut = |ctx: &egui::Context, modifiers: Modifiers, key: Key| {
            ctx.input_mut(|i| i.consume_shortcut(&KeyboardShortcut::new(modifiers, key)))
        };

        if ui.button("Reset ").clicked() || shortcut(ctx, Modifiers::NONE, Key::R) {
            self.interpreter
                .initial_state(&self.circuit, &mut self.circuit_state);
            self.update_circuit_state_texture(gl);
        }

        if ui.button("Step ").clicked() || shortcut(ctx, Modifiers::NONE, Key::ArrowRight) {
            self.interpreter
                .step(&self.circuit, &mut self.circuit_state);
            self.update_circuit_state_texture(gl);
        }
    }
    fn custom_painting(&mut self, ui: &mut Ui) {
        let (width, height) = (self.circuit.image.width(), self.circuit.image.height());
        let tex_size = Vec2::new(width as f32, height as f32);

        let surface_size = ui.available_size();

        let (rect, response) = ui.allocate_exact_size(surface_size, Sense::drag());

        if let (true, Some(hover_pos)) = (
            response.contains_pointer(),
            ui.input(|i| i.pointer.hover_pos()),
        ) {
            // let zoom_factor = ui.input(|i| i.zoom_delta());
            let (mut zoom_factor, scroll_delta) =
                ui.input(|i| (i.zoom_delta(), i.smooth_scroll_delta.y));

            if zoom_factor == 1. && scroll_delta != 0. {
                let scroll_zoom_speed = ui.ctx().options(|opt| opt.input_options.scroll_zoom_speed);
                zoom_factor += scroll_delta * scroll_zoom_speed
            };

            if zoom_factor != 1. {
                let center = hover_pos - response.rect.min;
                self.camera.zoom_surface(zoom_factor, center, surface_size);
            }
        }

        // --- Circuit interact ---
        if let Some(pointer_pos) = response.interact_pointer_pos() {
            if response.dragged_by(PointerButton::Primary) {
                // Position inside the image rect (in points)
                let local_pos = pointer_pos - response.rect.min;

                let texel = self.camera.surface_to_texel(local_pos, surface_size);
                let (x, y) = (texel.x as u32, texel.y as u32);
                if texel.x < 0. || texel.y < 0. || width <= x || height <= y {
                    self.cursor = None;
                } else {
                    self.cursor = Some((x, y));
                }
            }
            if response.dragged_by(PointerButton::Secondary) {
                self.camera.position -=
                    response.drag_delta() * self.camera.texels_per_surface_pixel();
            }
        }
        // let mut scroll_delta = ui.input(|i| i.smooth_scroll_delta);
        // self.camera.position -= scroll_delta * self.camera.texels_per_surface_pixel();

        let uv_min = self.camera.surface_to_texel(Vec2::ZERO, surface_size) / tex_size;
        let uv_max = self.camera.surface_to_texel(surface_size, surface_size) / tex_size;
        let uv_size = uv_max - uv_min;

        // Clone locals so we can move them into the paint callback:
        let program = self.program;
        let vertex_array = self.vertex_array;
        let tex_image = self.tex_image;
        let tex_nets = self.tex_nets;
        let tex_state = self.tex_state;

        let target_net = self
            .cursor
            .and_then(|(x, y)| self.circuit.image.pixel(x, y).net())
            .unwrap_or(0);

        let callback = PaintCallback {
            rect,
            callback: Arc::new(egui_glow::CallbackFn::new(move |_info, painter| {
                let gl = painter.gl();

                use glow::*;

                unsafe {
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
                    gl.bind_texture(glow::TEXTURE_BUFFER, Some(tex_state));
                    gl.uniform_1_i32(
                        Some(&gl.get_uniform_location(program, "tex_state").unwrap()),
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

                    let pixel_size = uv_size / surface_size;
                    gl.uniform_2_f32(
                        Some(&gl.get_uniform_location(program, "pixel_size").unwrap()),
                        pixel_size.x,
                        pixel_size.y,
                    );

                    gl.uniform_1_u32(
                        Some(&gl.get_uniform_location(program, "target_net").unwrap()),
                        target_net,
                    );

                    gl.bind_vertex_array(Some(vertex_array));
                    gl.draw_arrays(glow::TRIANGLES, 0, 3);
                }
            })),
        };
        ui.painter().add(callback);
    }

    fn init_shadersnew(gl: &glow::Context) -> (glow::Program, glow::VertexArray) {
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

            (program, vertex_array)
        }
    }

    fn update_circuit_state_texture(&self, gl: &glow::Context) {
        unsafe {
            gl.bind_buffer(glow::TEXTURE_BUFFER, Some(self.buffer_state));
            gl.buffer_sub_data_u8_slice(
                glow::TEXTURE_BUFFER,
                0,
                bytemuck::cast_slice(&self.circuit_state),
            );
        }
    }
}
