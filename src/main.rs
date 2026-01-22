use eframe::{
    egui::{
        self, CollapsingHeader, Key, KeyboardShortcut, Modifiers, PaintCallback, PointerButton,
        RichText, ScrollArea, Sense, Ui,
    },
    egui_glow,
    glow::{self},
};
use image::{ImageFormat, ImageReader};
use native_dialog::DialogBuilder;
use smallvec::SmallVec;
use std::{
    cell::Cell,
    collections::BTreeMap,
    fmt::{self, Debug, Display},
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

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
    circuit_playground: Option<CircuitPlayground>,

    cursor: Option<(u32, u32)>,

    current_folder: PathBuf,
    current_files: Vec<(String, PathBuf)>,
    load_circuit_error_message: Option<String>,

    fallback_file_dialog: Option<egui_file_dialog::FileDialog>,

    /// List of engines to try for each loaded circuit
    engines: Vec<Box<dyn CircuitEngine>>,
}

struct CircuitPlayground {
    name: String,
    path: PathBuf,
    circuit: CircuitImage,
    circuit_state: CircuitStateNets,

    /// Engine chosen to use for the circuit.
    engine: Box<dyn CircuitEngine>,

    /// Multiple engines are tested and mesured.
    /// The results hold the name and steps/second for each one of them.
    /// The first one are the results for the selected `self.engine`.
    benchmark_results: Vec<(&'static str, f32)>,
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
            circuit_playground: None,
            current_folder,
            current_files: Vec::new(),
            circuit_canvas,
            cursor: None,
            load_circuit_error_message: None,
            fallback_file_dialog: None,
            engines: vec![
                Box::new(CircuitEngineMadDfs::default()),
                Box::new(CircuitEngineUf::default()),
                Box::new(CircuitEngineUfT::default()),
                Box::new(CircuitEngineDfs::default()),
            ],
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
                self.show_circuit(ui, gl);
            });
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
            if ImageFormat::from_path(&path).is_ok() {
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
        self.circuit_playground = None;
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
        let mut circuit_state = CircuitStateNets::new(&circuit);

        self.circuit_canvas.load_circuit(gl, &circuit);
        self.circuit_canvas.load_circuit_state(gl, &circuit_state);

        let time_per_engine = Duration::from_millis(100);
        let mut benchmark_results: Vec<_> = self
            .engines
            .iter_mut()
            .map(|engine| (engine.name(), engine.bench(&circuit, time_per_engine)))
            .collect();

        benchmark_results.sort_unstable_by(|(_, a), (_, b)| b.total_cmp(a));

        let engine = self
            .engines
            .iter()
            .find(|e| e.name() == benchmark_results[0].0)
            .map_or_else::<Box<dyn CircuitEngine>, _, _>(
                || Box::new(default_engine(&circuit)),
                |e| e.new_dyn(&circuit),
            );

        circuit_state.reset();
        self.circuit_playground = Some(CircuitPlayground {
            name,
            path,
            circuit,
            circuit_state,
            engine,
            benchmark_results,
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
                    .circuit_playground
                    .as_ref()
                    .map(|r| r.name.clone())
                    .unwrap_or_else(|| "<Select Circuit>".into());

                let mut new_selection = None;
                egui::ComboBox::from_id_salt("LoadCircuit/combo")
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

        let Some(playground) = &self.circuit_playground else {
            return;
        };

        ui.strong(format!(
            "size: {:?} x {:?}",
            playground.circuit.width(),
            playground.circuit.height()
        ));
        ui.strong(format!("wires: {:?}", playground.circuit.wire_count() - 2));
        ui.strong(format!("gates: {:?}", playground.circuit.gate_count()));
    }

    fn choose_engine(&mut self, engine_name: &'static str) {
        let Some(playground) = &mut self.circuit_playground else {
            return;
        };

        let Some(engine) = self.engines.iter().find(|e| e.name() == engine_name) else {
            return;
        };
        playground.engine = engine.new_dyn(&playground.circuit);

        // Place selected engine result at top
        let Some(bech_pos) = playground
            .benchmark_results
            .iter()
            .position(|(n, _)| *n == engine_name)
        else {
            return;
        };

        playground.benchmark_results.swap(0, bech_pos);
        playground.benchmark_results[1..].sort_by(|(_, a), (_, b)| b.total_cmp(a));
    }

    fn show_selected_net_info(&mut self, ui: &mut Ui) {
        let Some((x, y)) = self.cursor else { return };
        let Some(playground) = &self.circuit_playground else {
            return;
        };
        let Some(&color) = playground.circuit.colors().get_pixel_checked(x, y) else {
            return;
        };

        self.separator(ui);
        ui.heading("Net Info");

        CollapsingHeader::new(RichText::new(format!("pixel  x: {x}  y: {y}")).strong())
            .id_salt("Net Info/pos/CollapsingHeader")
            .show(ui, |ui| {
                ui.strong(format!("rgb: {}, {}, {}", color[0], color[1], color[2]));
                ui.strong(format!("saturation: {:.0}%", 100. * hsv_saturation(color)));
                ui.strong(format!("value: {:.0}%", 100. * hsv_value(color)));
            });

        let pixel = playground.circuit.pixel(x, y);
        if let Some(net) = pixel.net() {
            ui.strong(format!("net: {:?}", net));

            if let Some(gate) = playground.circuit.get_gate(net) {
                ui.strong(format!("gate type: {:?}", gate.ty));
                ui.strong(format!("gate controls: {:?}", gate.controls));
                ui.strong(format!("gate wires: {:?}", gate.wires));
            } else {
                let gates = playground.circuit.connected_gates(net);
                ui.strong(format!("connected gates: {}", FmtIter::from(gates)));
            }
        }
    }

    fn show_execution_controls(&mut self, ctx: &egui::Context, ui: &mut Ui, gl: &glow::Context) {
        if self.circuit_playground.is_none() {
            return;
        }

        self.separator(ui);

        ui.heading("Execution");

        let Some(playground) = &mut self.circuit_playground else {
            return;
        };

        if let Some(&(best_name, best_result)) = playground.benchmark_results.first() {
            let current = format!("Engine: {} at {} steps/s", best_name, SiValue(best_result));
            if playground.benchmark_results.len() == 1 {
                ui.strong(current);
            } else {
                let mut selected_engine = None;
                CollapsingHeader::new(RichText::new(current).strong())
                    .id_salt("Circuit/BenchmarkResults/CollapsingHeader")
                    .show(ui, |ui| {
                        for &(name, result) in playground.benchmark_results.iter().skip(1) {
                            let resp =
                                ui.strong(format!("{} at {} steps/s", name, SiValue(result)));
                            if resp.clicked() {
                                selected_engine = Some(name);
                            }
                        }
                    });
                if let Some(engine_name) = selected_engine {
                    self.choose_engine(engine_name);
                }
            }
        }

        let Some(playground) = &self.circuit_playground else {
            return;
        };

        let shortcut = |ctx: &egui::Context, modifiers: Modifiers, key: Key| {
            ctx.input_mut(|i| i.consume_shortcut(&KeyboardShortcut::new(modifiers, key)))
        };

        if ui.button("Restart ").clicked() || shortcut(ctx, Modifiers::NONE, Key::R) {
            let camera = self.circuit_canvas.camera;
            let path = playground.path.clone();
            let previous_playground = self.circuit_playground.take();

            self.load_circuit(gl, path);
            self.load_circuit_error_message = None;

            // Restore state
            self.circuit_playground = self.circuit_playground.take().or(previous_playground);
            self.circuit_canvas.camera = camera;

            // Get playground mutably
            let Some(playground) = &mut self.circuit_playground else {
                return;
            };

            playground.circuit_state = CircuitStateNets::new(&playground.circuit);
            self.circuit_canvas
                .load_circuit_state(gl, &playground.circuit_state);
        }

        if ui.button("Step ").clicked()
            || shortcut(ctx, Modifiers::NONE, Key::ArrowRight)
            || shortcut(ctx, Modifiers::NONE, Key::S)
        {
            // Get playground mutably
            let Some(playground) = &mut self.circuit_playground else {
                return;
            };

            playground
                .engine
                .step(&playground.circuit, &mut playground.circuit_state);
            self.circuit_canvas
                .load_circuit_state(gl, &playground.circuit_state);
        }
    }

    fn show_circuit(&mut self, ui: &mut Ui, gl: &glow::Context) {
        let Some(playground) = &self.circuit_playground else {
            return;
        };

        // --- Allocate space for the circuit canvas ---
        let width = playground.circuit.width();
        let height = playground.circuit.height();
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

        // --- Click & Drag Interaction ---
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

        // --- Net Set/Unset ---
        let Some(playground) = &mut self.circuit_playground else {
            return;
        };
        if let Some((x, y)) = self.cursor
            && let Pixel::Wire { net, .. } = playground.circuit.pixel(x, y)
            && net != NET_OFF
            && net != NET_ON
        {
            if ui.input(|i| i.key_down(Key::Num1)) {
                playground.circuit_state[net as usize] = true;
                playground.circuit_state.update_gates(&playground.circuit);
                self.circuit_canvas
                    .load_circuit_state(gl, &playground.circuit_state);
            } else if ui.input(|i| i.key_down(Key::Num0)) {
                playground.circuit_state[net as usize] = false;
                playground.circuit_state.update_gates(&playground.circuit);
                self.circuit_canvas
                    .load_circuit_state(gl, &playground.circuit_state);
            }
        }

        self.circuit_canvas.selected_net = self
            .cursor
            .and_then(|(x, y)| playground.circuit.pixel(x, y).net())
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

struct FmtIter<I: IntoIterator>(Cell<Option<I>>);

impl<I: IntoIterator> From<I> for FmtIter<I> {
    fn from(value: I) -> Self {
        FmtIter(Cell::new(Some(value)))
    }
}

impl<I: IntoIterator> Display for FmtIter<I>
where
    <I as IntoIterator>::Item: Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Some(it) = self.0.replace(None) else {
            return write!(f, "[<consumed>]");
        };
        let mut it = it.into_iter();
        write!(f, "[")?;
        if let Some(first) = it.next() {
            write!(f, "{}", first)?;
            for x in it {
                write!(f, ", {}", x)?;
            }
        }
        write!(f, "]")?;
        Ok(())
    }
}

impl<I: IntoIterator> Debug for FmtIter<I>
where
    <I as IntoIterator>::Item: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Some(it) = self.0.replace(None) else {
            return write!(f, "[<consumed>]");
        };
        let mut it = it.into_iter();
        write!(f, "[")?;
        if let Some(first) = it.next() {
            write!(f, "{:?}", first)?;
            for x in it {
                write!(f, ", {:?}", x)?;
            }
        }
        write!(f, "]")?;
        Ok(())
    }
}
