use std::sync::Arc;

use eframe::{
    egui::{self, Key, PaintCallback, PointerButton, Sense, Ui, Vec2},
    egui_glow,
    glow::{self, HasContext},
};
use image::{EncodableLayout, ImageReader};

mod circuit;
use circuit::*;

fn main() {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([800.0, 600.0]),
        // multisampling: 4,
        renderer: eframe::Renderer::Glow,
        ..Default::default()
    };
    eframe::run_native(
        "My egui App",
        options,
        Box::new(|cc| Ok(Box::new(MyEguiApp::new(cc)))),
    )
    .unwrap();
}

struct MyEguiApp {
    tex_image: glow::NativeTexture,
    tex_nets: glow::NativeTexture,
    circuit: CircuitImage,
    cursor: (u32, u32),

    program: glow::Program,
    vertex_array: glow::VertexArray,
}

impl MyEguiApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let gl = cc.gl.as_ref().expect("Glow backend is needed");

        let img = ImageReader::open("circuits/test.png")
            .unwrap()
            .decode()
            .unwrap()
            .to_rgb8();

        let circuit = CircuitImage::new(img);

        let tex_image = unsafe {
            let texture = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));

            use glow::*;
            gl.tex_parameter_i32(TEXTURE_2D, TEXTURE_MIN_FILTER, NEAREST as i32);
            gl.tex_parameter_i32(TEXTURE_2D, TEXTURE_MAG_FILTER, NEAREST as i32);
            gl.tex_parameter_i32(TEXTURE_2D, TEXTURE_WRAP_S, CLAMP_TO_EDGE as i32);
            gl.tex_parameter_i32(TEXTURE_2D, TEXTURE_WRAP_T, CLAMP_TO_EDGE as i32);

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
                glow::PixelUnpackData::Slice(Some(circuit.image.as_bytes())),
            );
            texture
        };

        let tex_nets = unsafe {
            let texture = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));

            use glow::*;
            gl.tex_parameter_i32(TEXTURE_2D, TEXTURE_MIN_FILTER, NEAREST as i32);
            gl.tex_parameter_i32(TEXTURE_2D, TEXTURE_MAG_FILTER, NEAREST as i32);
            gl.tex_parameter_i32(TEXTURE_2D, TEXTURE_WRAP_S, CLAMP_TO_EDGE as i32);
            gl.tex_parameter_i32(TEXTURE_2D, TEXTURE_WRAP_T, CLAMP_TO_EDGE as i32);

            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);

            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::R32UI as i32,
                circuit.image.width() as i32,
                circuit.image.height() as i32,
                0,
                glow::RED_INTEGER,
                glow::UNSIGNED_INT,
                glow::PixelUnpackData::Slice(Some(bytemuck::cast_slice(&circuit.nets))),
            );
            texture
        };

        // let size = [img.width() as usize, img.height() as usize];
        // let pixels = img
        //     .chunks_exact(3)
        //     .map(|p| Color32::from_rgba_unmultiplied(p[0], p[1], p[2], 255))
        //     .collect();
        // let color_image = ColorImage::new(size, pixels);
        //
        // let texture: TextureHandle =
        //     cc.egui_ctx
        //         .load_texture("circuit.png", color_image, TextureOptions::NEAREST);

        let (program, vertex_array) = Self::init_shadersnew(gl);

        Self {
            tex_image,
            tex_nets,
            circuit,
            cursor: (0, 0),
            program,
            vertex_array,
        }
    }
}

impl eframe::App for MyEguiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if ctx.input(|i| i.key_pressed(Key::Escape)) {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }

        egui::SidePanel::left("left_bar")
            .resizable(false)
            .show(ctx, |ui| {
                ui.heading("Tools?");
                ui.separator();

                egui::ScrollArea::vertical()
                    .auto_shrink([false; 2])
                    .show(ui, |ui| {
                        let (x, y) = self.cursor;
                        ui.strong(format!("pos:   {x}, {y}"));
                        ui.strong(format!("net: {}", self.circuit.get_net(x, y)));

                        let color = *self.circuit.image.get_pixel(x, y);
                        ui.strong(format!("color: {color:?}"));
                        ui.strong(format!("saturation: {:.0}%", 100. * saturation(color)));
                        ui.strong(format!("value: {:.0}%", 100. * value(color)));

                        ui.separator();

                        ui.strong(format!("net count: {:?}", self.circuit.net_count));
                    });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::Frame::new().show(ui, |ui| {
                self.custom_painting(ui);
            });
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
    fn custom_painting(&mut self, ui: &mut Ui) {
        let tex_size = Vec2::new(
            self.circuit.image.width() as f32,
            self.circuit.image.height() as f32,
        );

        let max = ui.available_size();

        let rect_size = if tex_size.x / tex_size.y >= max.x / max.y {
            Vec2::new(max.x, tex_size.y * max.x / tex_size.x)
        } else {
            Vec2::new(tex_size.x * max.y / tex_size.y, max.y)
        };

        let (rect, response) = ui.allocate_exact_size(rect_size, Sense::drag());

        // Clone locals so we can move them into the paint callback:

        let program = self.program;
        let vertex_array = self.vertex_array;
        let tex_image = self.tex_image;
        let tex_nets = self.tex_nets;
        let target_net = self.circuit.get_net(self.cursor.0, self.cursor.1);

        let callback = PaintCallback {
            rect,
            callback: Arc::new(egui_glow::CallbackFn::new(move |_info, painter| {
                let gl = painter.gl();
                let resolution = rect_size;

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

                    gl.uniform_2_f32(
                        Some(&gl.get_uniform_location(program, "resolution").unwrap()),
                        resolution.x,
                        resolution.y,
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

        // --- Circuit interact ---
        if response.dragged_by(PointerButton::Primary)
            && let Some(pointer_pos) = response.interact_pointer_pos()
        {
            // Position inside the image rect (in points)
            let local_pos = pointer_pos - response.rect.min;

            // Normalize to UV coordinates (0.0–1.0)
            let uv = local_pos / response.rect.size();
            let uv = uv.clamp(Vec2::ZERO, Vec2::splat(1.0));

            // Step 3: convert UV to pixel coordinates
            let pixel_x = ((uv.x * tex_size.x).floor() as u32).min(tex_size.x as u32 - 1);
            let pixel_y = ((uv.y * tex_size.y).floor() as u32).min(tex_size.y as u32 - 1);
            self.cursor = (pixel_x, pixel_y);
        }
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
}
