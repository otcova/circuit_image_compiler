use eframe::{
    egui::{
        self, CollapsingHeader, Color32, DragValue, Event, Key, Modifiers, PaintCallback,
        PointerButton, RichText, ScrollArea, Sense, Theme, Ui, Vec2,
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
use tokio::sync::oneshot;

use circuit::*;
use circuit_canvas::*;

use crate::{
    circuit::{
        adder::{AdderHalt, CircuitEnvAdder, CircuitEnvAdderConfig},
        collatz::{CircuitEnvCollatz, CircuitEnvCollatzConfig, CollatzHalt},
        playground::CircuitEnvPlayground,
    },
    utils::{
        num_display::{GroupedUInt, SiValue},
        sync_state::SyncOutcome,
    },
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

enum Runner {
    Playground(CircuitRunner<CircuitEnvPlayground>),
    Adder(CircuitRunner<CircuitEnvAdder>),
    Collatz(CircuitRunner<CircuitEnvCollatz>),
}

impl Runner {
    pub fn as_ref(&self) -> &dyn CircuitRunnerTrait {
        match self {
            Runner::Playground(r) => r,
            Runner::Adder(r) => r,
            Runner::Collatz(r) => r,
        }
    }
    pub fn as_mut(&mut self) -> &mut dyn CircuitRunnerTrait {
        match self {
            Runner::Playground(r) => r,
            Runner::Adder(r) => r,
            Runner::Collatz(r) => r,
        }
    }
    pub fn circuit(&self) -> &CircuitState {
        self.as_ref().circuit()
    }
}

struct Playground {
    circuit_name: String,
    path: PathBuf,
    on_circuit_load: Option<oneshot::Receiver<(Runner, Camera)>>,
    runner: Option<Runner>,

    /// Multiple engines are tested and mesured.
    /// The results hold (the engine name, tps) sorted by tps (bigger first)
    benchmark_results: Vec<EngineBenchmarkResult>,
    benchmark_rx: Option<std::sync::mpsc::Receiver<EngineBenchmarkResult>>,
}

impl Playground {
    pub fn circuit(&self) -> Option<&CircuitState> {
        self.runner().map(|r| r.circuit())
    }

    pub fn runner(&self) -> Option<&dyn CircuitRunnerTrait> {
        self.runner.as_ref().map(|r| r.as_ref())
    }

    pub fn runner_mut(&mut self) -> Option<&mut dyn CircuitRunnerTrait> {
        self.runner.as_mut().map(|r| r.as_mut())
    }
}

impl MyEguiApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let mut dark_theme = egui::Visuals::dark();
        dark_theme.override_text_color = Some(Color32::from_gray(180));
        dark_theme.panel_fill = Color32::from_gray(15);
        cc.egui_ctx.set_visuals_of(Theme::Dark, dark_theme);

        let mut light_theme = egui::Visuals::light();
        light_theme.override_text_color = Some(Color32::from_gray(40));
        light_theme.panel_fill = Color32::from_gray(230);
        light_theme.window_fill = Color32::from_gray(230);
        light_theme.widgets.noninteractive.weak_bg_fill = Color32::from_gray(230);
        light_theme.widgets.noninteractive.bg_fill = Color32::from_gray(230);
        light_theme.widgets.inactive.weak_bg_fill = Color32::from_gray(210);
        light_theme.widgets.inactive.bg_fill = Color32::from_gray(210);
        light_theme.widgets.hovered.weak_bg_fill = Color32::from_gray(190);
        light_theme.widgets.hovered.bg_fill = Color32::from_gray(190);
        light_theme.widgets.active.weak_bg_fill = Color32::from_gray(150);
        light_theme.widgets.active.bg_fill = Color32::from_gray(150);
        light_theme.widgets.open.weak_bg_fill = Color32::from_gray(190);
        light_theme.widgets.open.bg_fill = Color32::from_gray(190);
        cc.egui_ctx.set_visuals_of(Theme::Light, light_theme);

        // // TODO: Allow user to configure visuals
        // cc.egui_ctx.set_theme(Theme::Light);
        // cc.egui_ctx.set_theme(Theme::Dark);

        let gl = cc.gl.as_ref().expect("Glow backend is needed");

        let current_folder = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let default_circuit = current_folder.join("circuits/playground.png");

        let circuit_canvas = CircuitCanvas::new(gl);

        // Configure UI Visuals
        cc.egui_ctx.style_mut(|style| {
            style.visuals.handle_shape = egui::style::HandleShape::Rect { aspect_ratio: 0.5 };
            style.spacing.scroll.bar_width = 8.;
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
        // Do at least 15fps when running and 1fps when not.
        if let Some(playground) = &self.playground
            && let Some(runner) = playground.runner()
            && !runner.is_paused()
        {
            ctx.request_repaint_after_secs(1. / 15.);
        } else {
            ctx.request_repaint_after_secs(1.);
        }

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

        if let Some(playground) = &mut self.playground {
            if let Some(circuit_rx) = &mut playground.on_circuit_load {
                match circuit_rx.try_recv() {
                    Ok((runner, camera)) => {
                        let circuit = &runner.circuit().image;
                        self.circuit_canvas.load_circuit(gl, circuit, camera);

                        playground.runner = Some(runner);
                        playground.on_circuit_load = None;
                    }
                    Err(oneshot::error::TryRecvError::Empty) => {}
                    Err(oneshot::error::TryRecvError::Closed) => {
                        // Failed to load the circuit
                        playground.runner = None;
                        playground.on_circuit_load = None;
                    }
                }
            }

            if let Some(runner) = &mut playground.runner
                && runner.as_mut().update() != SyncOutcome::NoChanges
            {
                self.circuit_canvas.load_circuit_state(gl, runner.circuit());
            }
        }

        egui::SidePanel::left("left_bar")
            .min_width(250.0)
            .frame(egui::Frame::new().fill(ctx.style().visuals.panel_fill))
            .resizable(false)
            .show(ctx, |ui| {
                ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        egui::Frame::new()
                            .inner_margin(egui::Margin::same(20))
                            .show(ui, |ui| {
                                self.show_circuit_info(ui);
                                self.show_execution_controls(ctx, ui);
                                self.show_environment(ui);
                                self.show_selected_net_info(ui);
                            });
                    });
            });

        egui::CentralPanel::default()
            .frame(egui::Frame::new())
            .show(ctx, |ui| {
                self.show_circuit(ui);
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

        // Load images in folder
        for path in fs::read_dir(folder)?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_file())
        {
            if ImageFormat::from_path(&path).is_ok() {
                let name = path
                    .file_stem()
                    .map(|s| format!("{}", s.display()))
                    .or_else(|| path.file_name().map(|n| format!("{}", n.display())))
                    .unwrap_or_else(|| "circuit".into());

                files.entry((0, name)).or_default().push(path);
            }
        }

        // Load images in folder/local
        if let Ok(local_folder) = fs::read_dir(folder.join("local")) {
            for path in local_folder
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.is_file())
            {
                if ImageFormat::from_path(&path).is_ok() {
                    let name = path
                        .file_stem()
                        .map(|s| format!("local/{}", s.display()))
                        .or_else(|| path.file_name().map(|n| format!("local/{}", n.display())))
                        .unwrap_or_else(|| "local/circuit".into());

                    files.entry((1, name)).or_default().push(path);
                }
            }
        }

        // Update current files
        for ((_, name), mut paths) in files {
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
        let path = path.into();

        // We store current camera in case of only reloading current circuit
        let previous_camera = if let Some(playground) = &self.playground
            && playground.path == path
        {
            Some(self.circuit_canvas.camera)
        } else {
            None
        };

        self.load_circuit_error_message = None;

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
                    let _ = self.open_folder(&self.current_folder.clone());
                    self.load_circuit_error_message = Some(format!(
                        "Unable to load the circuit: {}\n{}",
                        path.display(),
                        error
                    ));
                    return;
                }
            },

            Err(error) => {
                let _ = self.open_folder(&self.current_folder.clone());
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

        let (on_runner_load_tx, on_circuit_load) = oneshot::channel();
        let (benchmark_tx, benchmark_rx) = std::sync::mpsc::sync_channel(4);

        if previous_camera.is_some()
            && let Some(playground) = &mut self.playground
        {
            playground.on_circuit_load = Some(on_circuit_load);
            playground.benchmark_rx = Some(benchmark_rx);
        } else {
            self.playground = Some(Playground {
                circuit_name,
                path,
                runner: None,
                on_circuit_load: Some(on_circuit_load),

                benchmark_results: Vec::new(),
                benchmark_rx: Some(benchmark_rx),
            });
        }

        let runner_type = self
            .playground
            .as_ref()
            .and_then(|p| p.runner())
            .map(|r| r.env().name());

        self.rt.spawn(async move {
            let circuit = Arc::new(CircuitImage::new(img));
            let engine = Box::new(default_engine(&circuit));

            let camera = previous_camera.unwrap_or_else(|| {
                let tex_size = Vec2::new(circuit.width() as f32, circuit.height() as f32);
                let mut camera = Camera::new();
                camera.position = tex_size / 2.;
                camera.set_surface_pixels_per_texel(500. / tex_size.y);
                camera
            });

            let send_result = match runner_type {
                Some(CircuitEnvAdder::NAME) => {
                    let config = CircuitEnvAdderConfig::new(&circuit);
                    let env = CircuitEnvAdder::new(circuit.clone(), config);
                    let mut runner = CircuitRunner::new(env, engine);
                    runner.runtime.tick_interval = Duration::from_secs_f32(1. / 20.);
                    let runner = Runner::Adder(runner);
                    on_runner_load_tx.send((runner, camera))
                }
                Some(CircuitEnvCollatz::NAME) => {
                    let config = CircuitEnvCollatzConfig::new(&circuit);
                    let env = CircuitEnvCollatz::new(circuit.clone(), config);
                    let mut runner = CircuitRunner::new(env, engine);
                    runner.runtime.tick_interval = Duration::from_secs_f32(1. / 20.);
                    let runner = Runner::Collatz(runner);
                    on_runner_load_tx.send((runner, camera))
                }
                _ => {
                    let env = CircuitEnvPlayground::new(circuit.clone());
                    let mut runner = CircuitRunner::new(env, engine);
                    runner.runtime.tick_interval = Duration::from_secs_f32(1. / 20.);
                    let runner = Runner::Playground(runner);
                    on_runner_load_tx.send((runner, camera))
                }
            };

            if send_result.is_err() {
                return;
            }

            let mut engines = all_engines(&circuit);
            let mut state = CircuitState::new(circuit);

            let time_per_bench = Duration::from_millis(300);

            for engine in &mut engines {
                state.reset();
                let bench = engine.bench_tps(&mut state, time_per_bench);
                if benchmark_tx.send(bench).is_err() {
                    return;
                }
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
                        runner: circuit,
                        ..
                    }) => {
                        if circuit.is_some() {
                            circuit_name.clone()
                        } else {
                            format!("{circuit_name} (Loading)")
                        }
                    }
                };

                let mut new_selection = None;
                egui::ComboBox::from_id_salt("LoadCircuit/ComboBox")
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

        let Some(runner) = self.playground.as_mut().and_then(|p| p.runner.as_ref()) else {
            return;
        };
        let image = &runner.circuit().image;
        ui.label(format!("size: {:?} x {:?}", image.width(), image.height()));
        ui.label(format!("wires: {:?}", image.wire_count() - 2));
        ui.label(format!("gates: {:?}", image.gate_count()));

        let inp = image.inputs();
        let out = image.outputs();
        ui.label(format!("{} inputs: {:?}", inp.len(), inp));
        ui.label(format!("{} outputs: {:?}", out.len(), out));
    }

    fn choose_engine(&mut self, engine_name: &'static str) {
        let Some(playground) = &mut self.playground else {
            return;
        };
        let Some(runner) = &mut playground.runner_mut() else {
            return;
        };
        let Some(engine) = all_engines(&CircuitImage::empty())
            .into_iter()
            .find(|e| e.name() == engine_name)
        else {
            return;
        };

        runner.set_engine(engine.new_dyn(&runner.circuit().image));
        runner.publish();
    }

    fn show_selected_net_info(&mut self, ui: &mut Ui) {
        let Some((x, y)) = self.cursor else { return };
        let Some(circuit) = self.playground.as_ref().and_then(|p| p.circuit()) else {
            return;
        };

        let Some(&color) = circuit.image.colors().get_pixel_checked(x, y) else {
            return;
        };

        self.separator(ui);
        ui.heading("Net Info");

        CollapsingHeader::new(RichText::new(format!("pixel  x: {x}  y: {y}")))
            .id_salt("Net Info/pos/CollapsingHeader")
            .show(ui, |ui| {
                ui.label(format!("rgb: {}, {}, {}", color[0], color[1], color[2]));
                ui.label(format!("saturation: {:.0}%", 100. * hsv_saturation(color)));
                ui.label(format!("value: {:.0}%", 100. * hsv_value(color)));
            });

        let pixel = circuit.image.pixel(x, y);
        if let Some(net) = pixel.net() {
            ui.label(format!("net: {:?}", net));

            if let Some(gate) = circuit.image.get_gate(net) {
                ui.label(format!("gate type: {:?}", gate.ty));
                ui.label(format!("gate controls: {:?}", gate.controls));
                ui.label(format!("gate wires: {:?}", gate.wires));
            } else {
                let gates = circuit.image.connected_gates(net);
                ui.label(format!("connected gates: {}", FmtIter::from(gates)));

                let arrow = circuit.image.get_arrows(x, y);
                ui.label(format!("arrows: {}, {}", arrow.0, arrow.1));
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
        let Some(runner) = playground.runner() else {
            return;
        };

        self.separator(ui);

        ui.heading("Execution");

        let current_env_name = runner.env().name();
        let mut selected_env = current_env_name;

        egui::ComboBox::from_id_salt("Execution/Env/ComboBox")
            .selected_text(current_env_name)
            .show_ui(ui, |ui| {
                let mut add_item = |name| {
                    ui.selectable_value(&mut selected_env, name, name);
                };

                add_item(CircuitEnvPlayground::NAME);
                add_item(CircuitEnvAdder::NAME);
                add_item(CircuitEnvCollatz::NAME);
            });

        if selected_env != current_env_name
            && let Some(playground) = &mut self.playground
            && let Some(runner) = playground.runner()
        {
            let circuit = runner.circuit().image.clone();
            let engine = runner.engine().clone_dyn();
            let tick_interval = runner.tick_interval();

            match selected_env {
                CircuitEnvAdder::NAME => {
                    let config = CircuitEnvAdderConfig::new(&circuit);
                    let env = CircuitEnvAdder::new(circuit, config);
                    let mut runner = CircuitRunner::new(env, engine);
                    runner.runtime.tick_interval = tick_interval;
                    playground.runner = Some(Runner::Adder(runner));
                }
                CircuitEnvCollatz::NAME => {
                    let config = CircuitEnvCollatzConfig::new(&circuit);
                    let env = CircuitEnvCollatz::new(circuit, config);
                    let mut runner = CircuitRunner::new(env, engine);
                    runner.runtime.tick_interval = tick_interval;
                    playground.runner = Some(Runner::Collatz(runner));
                }
                _ => {
                    let env = CircuitEnvPlayground::new(circuit);
                    let mut runner = CircuitRunner::new(env, engine);
                    runner.runtime.tick_interval = tick_interval;
                    playground.runner = Some(Runner::Playground(runner));
                }
            }
        }
        let Some(playground) = &self.playground else {
            return;
        };
        let Some(runner) = playground.runner() else {
            return;
        };

        ui.label(format!("Tick: {}", GroupedUInt(runner.circuit().tick)));

        let mut selected_engine = None;
        let current_engine_name = runner.engine().name();

        let header = if let Some(current_bench) = playground
            .benchmark_results
            .iter()
            .find(|b| b.engine_name == current_engine_name)
        {
            format!("Engine TPS: {}", SiValue(current_bench.tps))
        } else {
            "Engine TPS: ...".into()
        };

        CollapsingHeader::new(RichText::new(header))
            .id_salt("Circuit/BenchmarkResults/CollapsingHeader")
            .show(ui, |ui| {
                for benchmark in playground.benchmark_results.iter() {
                    let resp = if benchmark.engine_name == current_engine_name {
                        ui.strong(format!("{}", benchmark))
                    } else {
                        ui.weak(format!("{}", benchmark))
                    };
                    if resp.clicked() {
                        selected_engine = Some(benchmark.engine_name);
                    }
                }
            });

        if let Some(engine_name) = selected_engine {
            self.choose_engine(engine_name);
        }

        let pressed_once = |ctx: &egui::Context, desired_key: Key| {
            ctx.input(|i| {
                i.events.iter().any(|event| {
                    matches!(
                        event,
                        Event::Key { key, modifiers: Modifiers::NONE, pressed: true, repeat: false, .. }
                        if *key == desired_key
                    )
                })
            })
        };

        // Restart will clear the circuit, we need to draw buttons before that.
        let restart = ui.button("Restart ").clicked() || pressed_once(ctx, Key::R);

        if let Some(playground) = &mut self.playground
            && let Some(runner) = playground.runner_mut()
        {
            if ui.button("Step ").clicked()
                || ctx.input_mut(|i| i.key_pressed(Key::ArrowRight))
                || ctx.input_mut(|i| i.key_pressed(Key::S))
            {
                runner.set_paused(true);
                runner.tick_n(1);
            }

            if runner.is_paused() {
                if ui.button("Play").clicked() || pressed_once(ctx, Key::Space) {
                    runner.set_paused(false);
                }
            } else if ui.button("Stop").clicked() || pressed_once(ctx, Key::Space) {
                runner.set_paused(true);
            }

            const MIN_TPS: f32 = 0.1;
            const MAX_TPS: f32 = 1_000_000.;
            let mut selected_tps = 1. / runner.tick_interval().as_secs_f32();
            let prev_tps = selected_tps;
            ui.add(
                egui::Slider::new(&mut selected_tps, MIN_TPS..=f32::INFINITY)
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
            if selected_tps != prev_tps {
                runner.set_tick_interval_secs(1. / selected_tps);
            }

            if restart {
                let path = playground.path.clone();
                self.load_circuit(path);
            }
        }
    }

    fn show_environment(&mut self, ui: &mut Ui) {
        let Some(playground) = &self.playground else {
            return;
        };
        let Some(runner) = &playground.runner else {
            return;
        };
        match runner {
            Runner::Playground(_) => {}
            Runner::Adder(runner) => {
                let env = &runner.runtime.env;
                let mut config = *env.config();

                self.separator(ui);
                ui.heading("Adder");

                egui::Grid::new("Env/Adder/IO/Config")
                    .num_columns(2)
                    .show(ui, |ui| {
                        ui.label("Input A bits");
                        ui.add(DragValue::new(&mut config.bits_inp_a).range(1..=128));
                        ui.end_row();

                        ui.label("Input B bits");
                        ui.add(DragValue::new(&mut config.bits_inp_b).range(1..=128));
                        ui.end_row();

                        ui.label("Output Sum bits");
                        ui.add(DragValue::new(&mut config.bits_out).range(1..=128));
                        ui.end_row();

                        ui.label("Operations");
                        ui.add(DragValue::new(&mut config.max_operations).range(1..=u32::MAX));
                        ui.end_row();

                        ui.label("Seed");
                        ui.add(DragValue::new(&mut config.seed));
                        ui.end_row();
                    });
                ui.add_space(10.);

                if config != *env.config()
                    && let Some(playground) = &mut self.playground
                    && let Some(Runner::Adder(runner)) = &mut playground.runner
                {
                    let circuit = runner.circuit().image.clone();
                    runner.runtime.env = CircuitEnvAdder::new(circuit, config);
                }

                let Some(playground) = &self.playground else {
                    return;
                };
                let Some(Runner::Adder(runner)) = &playground.runner else {
                    return;
                };
                let env = &runner.runtime.env;

                if let Ok(io) = env.get_io() {
                    ui.label(format!("input A: {}", io.inp_a));
                    ui.label(format!("input B: {}", io.inp_b));
                    ui.label(format!("output: {}", io.out));
                    ui.add_space(4.);
                    ui.label(format!("output enable: {}", io.next));
                    ui.label(format!("input enable: {}", io.done));
                    ui.add_space(4.);

                    ui.label(format!(
                        "Operations Done: {} / {}",
                        env.operations_done(),
                        config.max_operations
                    ));
                    CollapsingHeader::new(RichText::new(format!(
                        "Enqueued Operations: {}",
                        env.queue().len()
                    )))
                    .id_salt("Env/Adder/Queue/CollapsingHeader")
                    .show(ui, |ui| {
                        for item in env.queue().iter() {
                            ui.label(format!("{} + {} = {}", item.a, item.b, item.sum));
                        }
                    });

                    ui.add_space(4.);
                }

                match env.is_halt() {
                    Some(AdderHalt::Success) => {
                        ui.colored_label(Color32::GREEN, "Success");
                    }
                    Some(AdderHalt::InvalidIo) => {
                        ui.colored_label(Color32::ORANGE, "Invalid circuit");
                        ui.colored_label(
                            Color32::ORANGE,
                            format!(
                                "Expected {} inputs and {} outputs",
                                env.config().input_count(),
                                env.config().output_count()
                            ),
                        );
                    }
                    Some(AdderHalt::WrongOut {
                        a,
                        b,
                        expected,
                        got,
                    }) => {
                        ui.colored_label(Color32::RED, format!("Wrong output for {a} + {b}"));
                        ui.colored_label(
                            Color32::RED,
                            format!(" - Expected: {a} + {b} = {expected}"),
                        );
                        ui.colored_label(Color32::RED, format!(" - Got: {a} + {b} = {got}"));
                    }
                    None => {}
                }
            }
            Runner::Collatz(runner) => {
                let env = &runner.runtime.env;
                let mut config = *env.config();

                self.separator(ui);
                ui.heading("Collatz Steps (3n+1)");

                egui::Grid::new("Env/Adder/IO/Config")
                    .num_columns(2)
                    .show(ui, |ui| {
                        ui.label("Input bits");
                        ui.add(DragValue::new(&mut config.bits_inp).range(1..=128));
                        ui.end_row();

                        ui.label("Output bits");
                        ui.add(DragValue::new(&mut config.bits_out).range(1..=128));
                        ui.end_row();

                        ui.label("Operations");
                        ui.add(DragValue::new(&mut config.max_operations).range(1..=u32::MAX));
                        ui.end_row();

                        ui.label("Seed");
                        ui.add(DragValue::new(&mut config.seed));
                        ui.end_row();
                    });
                ui.add_space(10.);

                if config != *env.config()
                    && let Some(playground) = &mut self.playground
                    && let Some(Runner::Collatz(runner)) = &mut playground.runner
                {
                    let circuit = runner.circuit().image.clone();
                    runner.runtime.env = CircuitEnvCollatz::new(circuit, config);
                }

                let Some(playground) = &self.playground else {
                    return;
                };
                let Some(Runner::Collatz(runner)) = &playground.runner else {
                    return;
                };
                let env = &runner.runtime.env;

                if let Ok(io) = env.get_io() {
                    ui.label(format!("input: {}", io.inp));
                    ui.label(format!("output: {}", io.out));
                    ui.add_space(4.);
                    ui.label(format!("output enable: {}", io.next));
                    ui.label(format!("input enable: {}", io.done));
                    ui.add_space(4.);

                    ui.label(format!(
                        "Operations Done: {} / {}",
                        env.operations_done(),
                        config.max_operations
                    ));
                    CollapsingHeader::new(RichText::new(format!(
                        "Enqueued Operations: {}",
                        env.queue().len()
                    )))
                    .id_salt("Env/Collatz Steps/Queue/CollapsingHeader")
                    .show(ui, |ui| {
                        for item in env.queue().iter() {
                            ui.label(format!("{} -> {}", item.input, item.steps));
                        }
                    });

                    ui.add_space(4.);
                }

                match env.is_halt() {
                    Some(CollatzHalt::Success) => {
                        ui.colored_label(Color32::GREEN, "Success");
                    }
                    Some(CollatzHalt::InvalidIo) => {
                        ui.colored_label(Color32::ORANGE, "Invalid circuit");
                        ui.colored_label(
                            Color32::ORANGE,
                            format!(
                                "Expected {} inputs and {} outputs",
                                env.config().input_count(),
                                env.config().output_count()
                            ),
                        );
                    }
                    Some(CollatzHalt::WrongOut {
                        input,
                        expected,
                        got,
                    }) => {
                        ui.colored_label(Color32::RED, format!("Wrong output for {input}"));
                        ui.colored_label(
                            Color32::RED,
                            format!(" - Expected: {input} -> {expected}"),
                        );
                        ui.colored_label(Color32::RED, format!(" - Got: {input} -> {got}"));
                    }
                    None => {}
                }
            }
        }
    }

    fn show_circuit(&mut self, ui: &mut Ui) {
        let Some(playground) = &self.playground else {
            return;
        };
        let Some(runner) = &playground.runner else {
            return;
        };

        // --- Allocate space for the circuit canvas ---
        let width = runner.circuit().image.width();
        let height = runner.circuit().image.height();
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
            && let Pixel::Wire { net, .. } = runner.circuit().image.pixel(x, y)
            && let Some(playground) = &mut self.playground
            && let Some(runner) = playground.runner_mut()
            && net != NET_OFF
            && net != NET_ON
        {
            let inputs = runner.circuit().nets.inputs();

            if !inputs[net as usize] && ui.input(|i| i.key_down(Key::Num1)) {
                runner.env_mut().set_input(net, true);
            } else if inputs[net as usize] && ui.input(|i| i.key_down(Key::Num0)) {
                runner.env_mut().set_input(net, false);
            }
        }

        let Some(playground) = &self.playground else {
            return;
        };
        let Some(runner) = &playground.runner else {
            return;
        };

        self.circuit_canvas.selected_net = self
            .cursor
            .and_then(|(x, y)| runner.circuit().image.pixel(x, y).net())
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
