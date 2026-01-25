use core::f32;
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
    sync::{Arc, mpsc::TryRecvError},
    time::Duration,
};

use circuit::*;
use circuit_canvas::*;

use crate::utils::{
    num_display::{GroupedUInt, SiValue},
    promise::Promise,
};

mod circuit;
mod circuit_canvas;
mod utils;

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
    playground: Option<Playground>,

    cursor: Option<(u32, u32)>,

    current_folder: PathBuf,
    current_files: Vec<(String, PathBuf)>,
    load_circuit_error_message: Option<String>,

    fallback_file_dialog: Option<egui_file_dialog::FileDialog>,

    rt: tokio::runtime::Runtime,
}

struct Playground {
    circuit_name: String,
    path: PathBuf,
    runner: Promise<CircuitRunner>,
    state: Option<CircuitState>,
    engine_name: Option<&'static str>,

    /// ticks per second selected by the user
    target_tps: f32,

    /// Multiple engines are tested and mesured.
    /// The results hold (the engine name, tps) sorted by tps (bigger first)
    benchmark_results: Vec<EngineBenchmarkResult>,
    benchmark_rx: Option<std::sync::mpsc::Receiver<EngineBenchmarkResult>>,
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
            playground: None,

            current_folder,
            current_files: Vec::new(),

            cursor: None,
            circuit_canvas,
            load_circuit_error_message: None,

            fallback_file_dialog: None,

            rt: tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .build()
                .unwrap(),
        };

        app.load_circuit(&default_circuit);
        app.load_circuit_error_message = None;

        app
    }
}

impl eframe::App for MyEguiApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        ctx.request_repaint();
        let gl = frame.gl().unwrap();

        // Close App?
        if ctx.input(|i| i.key_pressed(Key::Escape)) {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }

        // File dialog update
        if let Some(file_dialog) = &mut self.fallback_file_dialog {
            file_dialog.update(ctx);
            if let Some(path) = file_dialog.take_picked() {
                self.load_circuit(&path);
            }
        }

        if let Some(playground) = &mut self.playground
            && let Some(runner) = playground.runner.get()
        {
            runner.get(|runtime| {
                if let Some(state) = &mut playground.state {
                    state.clone_from(&runtime.state);
                } else {
                    // Case new circuit
                    // Since runtime starts paused, we do not need to quickly release the locked runner.
                    playground.state = Some(runtime.state.clone());
                    self.circuit_canvas.load_circuit(gl, &runtime.state.circuit);
                }
                playground.engine_name = Some(runtime.engine.name());
            });

            if let Some(state) = &playground.state {
                self.circuit_canvas.load_circuit_state(gl, state);
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
                        self.show_circuit_info(ui);
                        self.show_execution_controls(ctx, ui);
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
        let mut files = BTreeMap::<_, SmallVec<_, 2>>::new();
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

    fn load_circuit(&mut self, path: impl Into<PathBuf>) {
        self.playground = None;
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

        let circuit_name = path
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

        let (runner, runner_done) = Promise::new();
        let (benchmark_tx, benchmark_rx) = std::sync::mpsc::sync_channel(4);

        self.playground = Some(Playground {
            circuit_name,
            path,
            runner,
            state: None,
            engine_name: None,

            target_tps: f32::INFINITY,

            benchmark_results: Vec::new(),
            benchmark_rx: Some(benchmark_rx),
        });

        self.rt.spawn(async move {
            let circuit = Arc::new(CircuitImage::new(img));
            let engine = Box::new(default_engine(&circuit));
            let mut state = CircuitState::new(circuit.clone());
            let _ = runner_done.send(CircuitRunner::new(state.clone(), engine));

            let mut engines = all_engines(&circuit);

            let time_per_bench = Duration::from_millis(300);

            for engine in &mut engines {
                state.nets.reset();
                let bench = engine.bench_tps(&mut state, time_per_bench);
                let _ = benchmark_tx.send(bench);
            }
        });
    }

    fn show_circuit_picker(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
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
                    self.load_circuit(&path);

                    self.current_folder = path;
                    self.current_folder.pop();
                }
            }

            if self.current_files.len() > 1 {
                let current_selection = match &self.playground {
                    None => "<Select Circuit>".into(),
                    Some(Playground {
                        circuit_name,
                        runner,
                        ..
                    }) => {
                        if runner.is_done() {
                            circuit_name.clone()
                        } else {
                            format!("{circuit_name} (Loading)")
                        }
                    }
                };

                let mut new_selection = None;
                egui::ComboBox::from_id_salt("LoadCircuit/combo")
                    .selected_text(current_selection)
                    .show_ui(ui, |ui| {
                        for (name, path) in &self.current_files {
                            ui.selectable_value(&mut new_selection, Some(path), name);
                        }
                    });

                if let Some(path) = new_selection {
                    self.load_circuit(path.clone());
                }
            }
        });

        if let Some(error) = &self.load_circuit_error_message {
            ui.colored_label(egui::Color32::RED, error);
        }

        ui.add_space(2.);
    }

    /// UI that shows the net and other info of the selected pixel
    fn show_circuit_info(&mut self, ui: &mut Ui) {
        ui.heading("Circuit");

        self.show_circuit_picker(ui);

        let Some(runner) = self.playground.as_mut().and_then(|p| p.runner.get()) else {
            return;
        };

        ui.strong(format!(
            "size: {:?} x {:?}",
            runner.circuit().width(),
            runner.circuit().height()
        ));
        ui.strong(format!("wires: {:?}", runner.circuit().wire_count() - 2));
        ui.strong(format!("gates: {:?}", runner.circuit().gate_count()));
    }

    fn choose_engine(&mut self, engine_name: &'static str) {
        let Some(playground) = &mut self.playground else {
            return;
        };
        let Some(state) = &playground.state else {
            return;
        };
        let Some(engine) = all_engines(&CircuitImage::empty())
            .into_iter()
            .find(|e| e.name() == engine_name)
        else {
            return;
        };

        let Some(runner) = playground.runner.get() else {
            return;
        };

        runner.set_engine(engine.new_dyn(&state.circuit));
        playground.engine_name = Some(engine_name);
    }

    fn show_selected_net_info(&mut self, ui: &mut Ui) {
        let Some((x, y)) = self.cursor else { return };
        let Some(runner) = self.playground.as_ref().and_then(|p| p.runner.get()) else {
            return;
        };

        let circuit = runner.circuit();
        let Some(&color) = circuit.colors().get_pixel_checked(x, y) else {
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

        let pixel = circuit.pixel(x, y);
        if let Some(net) = pixel.net() {
            ui.strong(format!("net: {:?}", net));

            if let Some(gate) = circuit.get_gate(net) {
                ui.strong(format!("gate type: {:?}", gate.ty));
                ui.strong(format!("gate controls: {:?}", gate.controls));
                ui.strong(format!("gate wires: {:?}", gate.wires));
            } else {
                let gates = circuit.connected_gates(net);
                ui.strong(format!("connected gates: {}", FmtIter::from(gates)));
            }
        }
    }

    fn show_execution_controls(&mut self, ctx: &egui::Context, ui: &mut Ui) {
        // Load new benchmarks
        if let Some(playground) = &mut self.playground
            && let Some(benchmark_rx) = &mut playground.benchmark_rx
        {
            loop {
                match benchmark_rx.try_recv() {
                    Ok(bench) => {
                        if let Some(slot) = playground
                            .benchmark_results
                            .iter_mut()
                            .find(|b| b.engine_name == bench.engine_name)
                        {
                            *slot = bench;
                        } else {
                            playground.benchmark_results.push(bench);
                        }
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        playground.benchmark_rx = None;
                        break;
                    }
                }
            }

            // Sort them to display best one first.
            // (Ussing stable sort since it's displayed)
            playground
                .benchmark_results
                .sort_by(|a, b| b.tps.total_cmp(&a.tps));
        }

        let Some(playground) = &self.playground else {
            return;
        };
        let Some(state) = &playground.state else {
            return;
        };
        let Some(engine_name) = playground.engine_name else {
            return;
        };

        self.separator(ui);

        ui.heading("Execution");

        ui.strong(format!("Tick: {}", GroupedUInt(state.tick)));

        let mut selected_engine = None;

        let header = if let Some(current_bench) = playground
            .benchmark_results
            .iter()
            .find(|b| b.engine_name == engine_name)
        {
            format!("Engine TPS: {}", SiValue(current_bench.tps))
        } else {
            "Engine TPS: ...".into()
        };

        CollapsingHeader::new(RichText::new(header).strong())
            .id_salt("Circuit/BenchmarkResults/CollapsingHeader")
            .show(ui, |ui| {
                for benchmark in playground.benchmark_results.iter() {
                    let resp = ui.strong(format!("{}", benchmark));
                    if resp.clicked() {
                        selected_engine = Some(benchmark.engine_name);
                    }
                }
            });

        if let Some(engine_name) = selected_engine {
            self.choose_engine(engine_name);
        }

        let shortcut = |ctx: &egui::Context, modifiers: Modifiers, key: Key| {
            ctx.input_mut(|i| i.consume_shortcut(&KeyboardShortcut::new(modifiers, key)))
        };

        // Restart will clear the circuit, we need to draw buttons before that.
        let restart = ui.button("Restart ").clicked() || shortcut(ctx, Modifiers::NONE, Key::R);

        if let Some(playground) = &mut self.playground
            && let Some(runner) = &mut playground.runner.get_mut()
        {
            if ui.button("Step ").clicked()
                || shortcut(ctx, Modifiers::NONE, Key::ArrowRight)
                || shortcut(ctx, Modifiers::NONE, Key::S)
            {
                runner.tick_n(1);
            }

            if runner.is_paused() {
                if ui.button("Play").clicked() || shortcut(ctx, Modifiers::NONE, Key::Space) {
                    let dt = Duration::from_secs_f32(1. / playground.target_tps);
                    runner.set_tick_interval(Some(dt));
                }
            } else if ui.button("Stop").clicked() || shortcut(ctx, Modifiers::NONE, Key::Space) {
                runner.set_tick_interval(None);
            }

            const MIN_TPS: f32 = 0.1;
            const MAX_TPS: f32 = 1_000_000.;
            let prev_target_tps = playground.target_tps;
            ui.add(
                egui::Slider::new(&mut playground.target_tps, MIN_TPS..=f32::INFINITY)
                    .custom_formatter(|n, _| {
                        if n == f64::INFINITY {
                            "Unlimited".into()
                        } else {
                            SiValue(n as f32).to_string()
                        }
                    })
                    .text("Max TPS")
                    .largest_finite(MAX_TPS as f64)
                    .logarithmic(true),
            );
            if prev_target_tps != playground.target_tps && !runner.is_paused() {
                runner.set_tick_interval(Some(Duration::from_secs_f32(1. / playground.target_tps)));
            }

            if restart {
                let path = playground.path.clone();
                let camera = self.circuit_canvas.camera;
                self.load_circuit(path);
                self.circuit_canvas.camera = camera;
            }
        }
    }

    fn show_circuit(&mut self, ui: &mut Ui, gl: &glow::Context) {
        let Some(playground) = &self.playground else {
            return;
        };
        let Some(state) = &playground.state else {
            return;
        };

        // --- Allocate space for the circuit canvas ---
        let width = state.circuit.width();
        let height = state.circuit.height();
        let surface_size = ui.available_size();
        let (rect, response) = ui.allocate_exact_size(surface_size, Sense::drag());

        // --- Zoom Interaction ---
        if let (true, Some(hover_pos)) = (
            response.contains_pointer(),
            ui.input(|i| i.pointer.hover_pos()),
        ) {
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
        if let Some((x, y)) = self.cursor
            && let Pixel::Wire { net, .. } = state.circuit.pixel(x, y)
            && let Some(playground) = &mut self.playground
            && let Some(runner) = playground.runner.get_mut()
            && let Some(state) = &mut playground.state
            && net != NET_OFF
            && net != NET_ON
        {
            let inputs = state.nets.inputs_mut();
            if !inputs[net as usize] && ui.input(|i| i.key_down(Key::Num1)) {
                inputs[net as usize] = true;
                runner.overwrite(|r| r.state.clone_from(state));
                self.circuit_canvas.load_circuit_state(gl, state);
            } else if inputs[net as usize] && ui.input(|i| i.key_down(Key::Num0)) {
                inputs[net as usize] = false;
                runner.overwrite(|r| r.state.clone_from(state));
                self.circuit_canvas.load_circuit_state(gl, state);
            }
        }

        let Some(playground) = &self.playground else {
            return;
        };
        let Some(state) = &playground.state else {
            return;
        };

        self.circuit_canvas.selected_net = self
            .cursor
            .and_then(|(x, y)| state.circuit.pixel(x, y).net())
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

impl Display for EngineBenchmarkResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} tps  -  {}", SiValue(self.tps), self.engine_name)
    }
}
