use eframe::{
    egui::{
        self, CollapsingHeader, Key, KeyboardShortcut, Modifiers, PaintCallback, PointerButton,
        RichText, ScrollArea, Sense, Ui,
    },
    egui_glow,
    glow::{self, HasContext},
};
use image::ImageReader;
use native_dialog::DialogBuilder;
use smallvec::SmallVec;
use std::{
    collections::BTreeMap,
    fmt, fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use bench::*;
use circuit::*;
use circuit_canvas::*;

mod bench;
mod circuit;
mod circuit_canvas;

fn main() {
    env_logger::init();
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
    circuit_canvas: CircuitCanvas,
    circuit_runtime: Option<CircuitPlayground>,

    cursor: Option<(u32, u32)>,

    current_folder: PathBuf,
    current_files: Vec<(String, PathBuf)>,
    load_circuit_error_message: Option<String>,

    fallback_file_dialog: Option<egui_file_dialog::FileDialog>,
}

struct CircuitPlayground {
    name: String,
    path: PathBuf,
    circuit: CircuitImage,
    circuit_state: CircuitImageState,
    interpreter: CircuitInterpreterUF,
    /// Seconds that the circuit needs to run a step
    steps_per_second: f32,
}

impl MyEguiApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let gl = cc.gl.as_ref().expect("Glow backend is needed");

        let current_folder = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let default_circuit = current_folder.join("circuits/test.png");

        let circuit_canvas = CircuitCanvas::new(gl);

        // Configure UI Visuals
        cc.egui_ctx.style_mut(|style| {
            style.visuals.handle_shape = egui::style::HandleShape::Rect { aspect_ratio: 0.5 };
            style.spacing.scroll.bar_width = 6.;
            style.spacing.scroll.foreground_color = false;
            style.spacing.item_spacing.y = 4.;
        });

        let mut app = Self {
            circuit_runtime: None,
            current_folder,
            current_files: Vec::new(),
            circuit_canvas,
            cursor: None,
            load_circuit_error_message: None,
            fallback_file_dialog: None,
        };

        app.load_circuit(gl, &default_circuit);
        app.load_circuit_error_message = None;

        app
    }
}

impl eframe::App for MyEguiApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        let gl = frame.gl().unwrap();

        if ctx.input(|i| i.key_pressed(Key::Escape)) {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }

        if let Some(file_dialog) = &mut self.fallback_file_dialog {
            file_dialog.update(ctx);
            if let Some(path) = file_dialog.take_picked() {
                self.load_circuit(gl, &path);
            }
        }

        egui::SidePanel::left("left_bar")
            .min_width(250.0)
            .resizable(false)
            .frame(egui::Frame::new().inner_margin(egui::Margin::same(20)))
            .show(ctx, |ui| {
                ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        self.show_circuit_info(ui, gl);
                        self.show_execution_controls(ctx, ui, gl);
                        self.show_selected_net_info(ui);
                    });
            });

        egui::CentralPanel::default()
            .frame(egui::Frame::new())
            .show(ctx, |ui| {
                self.show_circuit(ui);
            });

        unsafe {
            gl.bind_buffer(glow::TEXTURE_BUFFER, None);
            gl.bind_texture(glow::TEXTURE_BUFFER, None);
            gl.bind_texture(glow::TEXTURE_2D, None);
            gl.use_program(None);
        }
    }

    fn on_exit(&mut self, gl: Option<&glow::Context>) {
        if let Some(gl) = gl {
            self.circuit_canvas.delete(gl);
        }
    }
}

impl MyEguiApp {
    fn separator(&self, ui: &mut Ui) {
        ui.add(egui::Separator::default().spacing(10.));
    }

    fn open_folder(&mut self, folder: &Path) -> std::io::Result<()> {
        self.current_folder = folder.into();

        // To check file name collisions.
        // BTree to sort files alphabetically for better UX
        let mut files = BTreeMap::<_, SmallVec<[_; 2]>>::new();
        self.current_files.clear();

        for path in fs::read_dir(folder)?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_file())
        {
            if let Some(ext) = path.extension()
                && ext.to_string_lossy().to_lowercase() == "png"
            {
                let name = path
                    .file_stem()
                    .map(|s| s.to_string_lossy().into())
                    .or_else(|| path.file_name().map(|n| n.to_string_lossy().into()))
                    .unwrap_or_else(|| "circuit".into());

                files.entry(name).or_default().push(path);
            }
        }

        // Update current files
        for (name, mut paths) in files {
            if paths.len() == 1
                && let Some(path) = paths.pop()
            {
                self.current_files.push((name, path));
                continue;
            }

            // Enumerate files with same name
            for (i, path) in paths.into_iter().enumerate() {
                self.current_files.push((format!("{name} ({i})"), path));
            }
        }

        Ok(())
    }

    fn load_circuit(&mut self, gl: &glow::Context, path: impl Into<PathBuf>) {
        self.circuit_runtime = None;
        self.load_circuit_error_message = None;

        let path = path.into();

        if path.is_dir() {
            if let Err(error) = self.open_folder(&path) {
                self.load_circuit_error_message = Some(format!(
                    "Unable to open the folder: {}\n{}",
                    path.display(),
                    error
                ));
            }
            return;
        }

        let name = path
            .file_stem()
            .map(|s| s.to_string_lossy().into())
            .unwrap_or_else(|| "circuit".into());

        let img = match ImageReader::open(&path) {
            Ok(img) => match img.decode() {
                Ok(img) => img.to_rgb8(),
                Err(error) => {
                    self.load_circuit_error_message = Some(format!(
                        "Unable to load the circuit: {}\n{}",
                        path.display(),
                        error
                    ));
                    return;
                }
            },

            Err(error) => {
                self.load_circuit_error_message = Some(format!(
                    "Unable to find the circuit: {}\n{}",
                    path.display(),
                    error
                ));
                return;
            }
        };

        if let Some(folder) = path.parent() {
            let _ = self.open_folder(folder);
        };

        let circuit = CircuitImage::new(img);
        let mut interpreter = CircuitInterpreterUF::default();
        let mut circuit_state = CircuitImageState::new(&circuit);

        self.circuit_canvas.load_circuit(gl, &circuit);
        self.circuit_canvas.load_circuit_state(gl, &circuit_state);

        let steps_per_second = 1.
            / bench_seconds(
                || interpreter.step(&circuit, &mut circuit_state),
                Duration::from_millis(100),
            );

        circuit_state.reset();
        self.circuit_runtime = Some(CircuitPlayground {
            name,
            path,
            circuit,
            circuit_state,
            interpreter,
            steps_per_second,
        });
    }

    fn show_circuit_picker(&mut self, ui: &mut Ui, gl: &glow::Context) {
        ui.horizontal(|ui| {
            // TODO: Folder image
            if ui.button("Load File").clicked() {
                // Open file picker (try native, if not fallback with egui)
                if let Some(path) = match DialogBuilder::file().open_single_file().show() {
                    Ok(path) => path,
                    Err(_) => {
                        let mut file_dialog = egui_file_dialog::FileDialog::new();
                        file_dialog.pick_file();
                        self.fallback_file_dialog = Some(file_dialog);
                        None
                    }
                } {
                    self.load_circuit(gl, &path);

                    self.current_folder = path;
                    self.current_folder.pop();
                }
            }

            if self.current_files.len() > 1 {
                let current_selection = self
                    .circuit_runtime
                    .as_ref()
                    .map(|r| r.name.clone())
                    .unwrap_or_else(|| "<Select Circuit>".into());

                let mut new_selection = None;
                egui::ComboBox::from_id_salt("png_combo")
                    .selected_text(current_selection)
                    .show_ui(ui, |ui| {
                        for (name, path) in &self.current_files {
                            ui.selectable_value(&mut new_selection, Some(path), name);
                        }
                    });

                if let Some(path) = new_selection {
                    self.load_circuit(gl, path.clone());
                }
            }
        });

        if let Some(error) = &self.load_circuit_error_message {
            ui.colored_label(egui::Color32::RED, error);
        }

        ui.add_space(2.);
    }

    /// UI that shows the net and other info of the selected pixel
    fn show_circuit_info(&mut self, ui: &mut Ui, gl: &glow::Context) {
        ui.heading("Circuit");

        self.show_circuit_picker(ui, gl);

        let Some(runtime) = &self.circuit_runtime else {
            return;
        };

        ui.strong(format!(
            "size: {:?} x {:?}",
            runtime.circuit.width(),
            runtime.circuit.height()
        ));
        ui.strong(format!("wires: {:?}", runtime.circuit.wire_count() - 2));
        ui.strong(format!("gates: {:?}", runtime.circuit.gate_count()));
        ui.strong(format!("steps/s: {}", SiValue(runtime.steps_per_second)));
    }

    fn show_selected_net_info(&mut self, ui: &mut Ui) {
        let Some((x, y)) = self.cursor else { return };
        let Some(runtime) = &self.circuit_runtime else {
            return;
        };
        let Some(&color) = runtime.circuit.colors().get_pixel_checked(x, y) else {
            return;
        };

        self.separator(ui);
        ui.heading("Net Info");

        CollapsingHeader::new(RichText::new(format!("pixel  x: {x}  y: {y}")).strong())
            .id_salt("Net Info/pos")
            .show(ui, |ui| {
                ui.strong(format!("rgb: {}, {}, {}", color[0], color[1], color[2]));
                ui.strong(format!("saturation: {:.0}%", 100. * hsv_saturation(color)));
                ui.strong(format!("value: {:.0}%", 100. * hsv_value(color)));
            });

        let pixel = runtime.circuit.pixel(x, y);
        if let Some(net) = pixel.net() {
            ui.strong(format!("net: {:?}", net));

            if let Some(gate) = runtime.circuit.get_gate(net) {
                ui.strong(format!("gate type: {:?}", gate.ty));
                ui.strong(format!("gate controls: {:?}", gate.controls));
                ui.strong(format!("gate wires: {:?}", gate.wires));
            }
        }
    }

    fn show_execution_controls(&mut self, ctx: &egui::Context, ui: &mut Ui, gl: &glow::Context) {
        let Some(runtime) = &self.circuit_runtime else {
            return;
        };

        self.separator(ui);

        ui.heading("Execution");

        let shortcut = |ctx: &egui::Context, modifiers: Modifiers, key: Key| {
            ctx.input_mut(|i| i.consume_shortcut(&KeyboardShortcut::new(modifiers, key)))
        };

        if ui.button("Restart ").clicked() || shortcut(ctx, Modifiers::NONE, Key::R) {
            let camera = self.circuit_canvas.camera;
            let path = runtime.path.clone();
            let previous_runtime = self.circuit_runtime.take();

            self.load_circuit(gl, path);
            self.load_circuit_error_message = None;

            // Restore state
            self.circuit_runtime = self.circuit_runtime.take().or(previous_runtime);
            self.circuit_canvas.camera = camera;

            // Get runtime mutably
            let Some(runtime) = &mut self.circuit_runtime else {
                return;
            };

            runtime.circuit_state = CircuitImageState::new(&runtime.circuit);
            self.circuit_canvas
                .load_circuit_state(gl, &runtime.circuit_state);
        }

        if ui.button("Step ").clicked()
            || shortcut(ctx, Modifiers::NONE, Key::ArrowRight)
            || shortcut(ctx, Modifiers::NONE, Key::S)
        {
            // Get runtime mutably
            let Some(runtime) = &mut self.circuit_runtime else {
                return;
            };

            runtime
                .interpreter
                .step(&runtime.circuit, &mut runtime.circuit_state);
            self.circuit_canvas
                .load_circuit_state(gl, &runtime.circuit_state);
        }
    }

    fn show_circuit(&mut self, ui: &mut Ui) {
        let Some(runtime) = &self.circuit_runtime else {
            return;
        };

        // --- Allocate space for the circuit canvas ---
        let width = runtime.circuit.width();
        let height = runtime.circuit.height();
        let surface_size = ui.available_size();
        let (rect, response) = ui.allocate_exact_size(surface_size, Sense::drag());

        // --- Zoom Interaction ---
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
                self.circuit_canvas
                    .camera
                    .zoom_surface(zoom_factor, center, surface_size);
            }
        }

        // --- Click/Drag Interaction ---
        if let Some(pointer_pos) = response.interact_pointer_pos() {
            if response.dragged_by(PointerButton::Primary) {
                // Position inside the image rect (in points)
                let local_pos = pointer_pos - response.rect.min;

                let texel = self
                    .circuit_canvas
                    .camera
                    .surface_to_texel(local_pos, surface_size);
                let (x, y) = (texel.x as u32, texel.y as u32);
                if texel.x < 0. || texel.y < 0. || width <= x || height <= y {
                    self.cursor = None;
                } else {
                    self.cursor = Some((x, y));
                }
            }

            if response.dragged_by(PointerButton::Secondary) {
                self.circuit_canvas.camera.position -=
                    response.drag_delta() * self.circuit_canvas.camera.texels_per_surface_pixel();
            }
        }

        self.circuit_canvas.selected_net = self
            .cursor
            .and_then(|(x, y)| runtime.circuit.pixel(x, y).net())
            .unwrap_or(0);

        // --- Draw Circuit ---
        let render_callback = self.circuit_canvas.render_callback(rect);
        let callback = PaintCallback {
            rect,
            callback: Arc::new(egui_glow::CallbackFn::new(move |_info, painter| {
                render_callback(painter.gl());
            })),
        };
        ui.painter().add(callback);
    }
}

struct SiValue(f32);

impl fmt::Display for SiValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const SIG_FIGS: i32 = 3;

        let value = self.0;
        let abs = value.abs();

        let prefixes = [(1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")];

        let (scaled, suffix) = prefixes
            .iter()
            .find(|(factor, _)| abs >= *factor)
            .map(|(factor, suffix)| (value / factor, *suffix))
            .unwrap_or((value, ""));

        // Number of digits before the decimal point
        let int_digits = scaled.abs().log10().floor() as i32 + 1;

        // Precision needed to reach SIG_FIGS
        let precision = (SIG_FIGS - int_digits).max(0) as usize;

        write!(f, "{:.*}{}", precision, scaled, suffix)
    }
}
