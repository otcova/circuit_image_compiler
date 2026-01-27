use image::{ImageBuffer, Rgb};
use smallvec::SmallVec;
use std::sync::Arc;

pub use circuit_runner::*;
pub use engine::*;

use crate::utils::{dense_map::DenseMap, union_find::UnionFind};

mod circuit_runner;
mod engine;

// Permanent unconditional off
pub const NET_OFF: u32 = 0;

// Permanent unconditional on (power)
pub const NET_ON: u32 = 1;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum GateType {
    Passive,
    Active,
}

impl GateType {
    /// Returns weather it connects (or not) the wires if all controls are set (if toggled).
    pub fn connects_wires(self, toggled: bool) -> bool {
        match self {
            GateType::Passive => !toggled,
            GateType::Active => toggled,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Pixel {
    Insulator,
    Power,
    Gate { ty: GateType, net: u32 },
    Wire { color: Rgb<u8>, net: u32 },
}

impl Pixel {
    pub fn net(self) -> Option<u32> {
        match self {
            Pixel::Insulator => None,
            Pixel::Power => Some(NET_ON),
            Pixel::Gate { net, .. } => Some(net),
            Pixel::Wire { net, .. } => Some(net),
        }
    }

    pub fn is_wire(self) -> bool {
        matches!(self, Pixel::Wire { .. })
    }
}

#[derive(Clone, Debug)]
pub struct Gate {
    pub ty: GateType,
    /// Gate-Gate contacts
    pub controls: SmallVec<u32, 4>,
    /// Gate-Wire contacts
    pub wires: SmallVec<u32, 4>,
}

#[derive(Clone, Debug)]
pub struct GateConnections {
    /// Gate-Gate contacts
    pub controls: SmallVec<u32, 4>,
    /// Gate-Wire contacts
    pub wires: SmallVec<u32, 4>,
}

impl Gate {
    pub fn new(ty: GateType) -> Gate {
        Gate {
            ty,
            controls: SmallVec::new(),
            wires: SmallVec::new(),
        }
    }
}

#[derive(Clone)]
pub struct CircuitImage {
    /// The circuit rgb image.
    colors: ImageBuffer<Rgb<u8>, Vec<u8>>,

    /// The net of each pixel of the image.
    image_nets: Vec<u32>,

    /// For each pixel, the horizontal and vertical arrows that where used
    /// to decide wire crossings.
    arrows: Vec<(u32, u32)>,

    /// Input Wire Nets
    inputs: Vec<u32>,

    /// Output Wire Nets
    outputs: Vec<u32>,

    /// Each wire stores the index of all the `self.gates` connected to it.
    wires: Vec<SmallVec<u32, 4>>,

    /// All gates from the circuit in net order, each element stores
    /// - the type of gate
    /// - the index of all the gates connected (that control the behaviour)
    /// - all the wires connected to it.
    pub gates: Vec<Gate>,

    /// The amount of gates that are a not or buffer gate.
    /// TODO: Also consider permanent on/off grates as trivial?.
    non_trivial_gates: Vec<Gate>,

    /// Each wire stores the index of all the `self.non_trivial_gates` connected.
    wires_non_trivial: Vec<SmallVec<u32, 4>>,

    /// All passive gates with NET_ON connected as a wire.
    /// This NET_ON is exclusded in the wires list of GateConnections.
    buffer_gates: Vec<GateConnections>,

    /// All active gates with NET_ON connected as a wire.
    /// This NET_ON is exclusded in the wires list of GateConnections.
    not_gates: Vec<GateConnections>,

    power_color: Rgb<u8>,
    active_gate_color: Rgb<u8>,
    passive_gate_color: Rgb<u8>,
}

pub fn hsv_value(pixel: Rgb<u8>) -> f32 {
    let (r, g, b) = (pixel[0], pixel[1], pixel[2]);
    r.max(g).max(b) as f32 / 255.0
}

pub fn hsv_saturation(pixel: Rgb<u8>) -> f32 {
    let (r, g, b) = (pixel[0], pixel[1], pixel[2]);

    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    if max == 0 {
        0.0
    } else {
        delta as f32 / max as f32
    }
}

impl CircuitImage {
    pub fn empty() -> Self {
        Self {
            colors: ImageBuffer::from_pixel(0, 0, Rgb([0, 0, 0])),
            image_nets: Vec::new(),
            arrows: Vec::new(),

            inputs: Vec::new(),
            outputs: Vec::new(),

            wires: Vec::new(),
            gates: Vec::new(),
            non_trivial_gates: Vec::new(),
            wires_non_trivial: Vec::new(),
            buffer_gates: Vec::new(),
            not_gates: Vec::new(),

            power_color: Rgb([255, 0, 0]),
            active_gate_color: Rgb([0, 255, 0]),
            passive_gate_color: Rgb([0, 0, 255]),
        }
    }

    // 20% value threshold for the colors that are considered insulators.
    // (Comparison is rounded down to 2 digits => add 0.5% to max)
    const INSULATOR_COLOR_MAX_VALUE: f32 = 0.205;

    /// Computes a score that represents how much the color matches the inteded color for
    /// (power, active gate, passive gate).
    fn color_match_score(color: Rgb<u8>) -> (i32, i32, i32) {
        let (r, g, b) = (color[0] as i32, color[1] as i32, color[2] as i32);
        (r - g.max(b), g - r.max(b), b - r.max(g))
    }

    /// Given a set of colors, it choses the ones that represent (power, active gate, passive gate)
    fn choose_matching_colors(
        colors: impl Iterator<Item = Rgb<u8>>,
    ) -> (Rgb<u8>, Rgb<u8>, Rgb<u8>) {
        // Initialize with lowerbound color match score.
        let mut power_max_score = 0;
        let mut active_gate_max_score = 0;
        let mut passive_gate_max_score = 0;

        // Configure default colors in case none is found with enouch match score.
        let mut power_color = Rgb([255, 0, 0]);
        let mut active_gate_color = Rgb([0, 255, 0]);
        let mut passive_gate_color = Rgb([0, 0, 255]);

        // Scan image to find power & gate colors
        for pixel in colors {
            let (power, active, passive) = Self::color_match_score(pixel);

            if power >= active && power >= passive {
                if power > power_max_score {
                    power_max_score = power;
                    power_color = pixel;
                }
            } else if active >= passive {
                if active > active_gate_max_score {
                    active_gate_max_score = active;
                    active_gate_color = pixel;
                }
            } else if passive > passive_gate_max_score {
                passive_gate_max_score = passive;
                passive_gate_color = pixel;
            }
        }

        (power_color, active_gate_color, passive_gate_color)
    }

    pub fn set_net(&mut self, x: u32, y: u32, net: u32) {
        let index = x as usize + y as usize * self.width() as usize;
        self.image_nets[index] = net;
    }

    pub fn colors(&self) -> &ImageBuffer<Rgb<u8>, Vec<u8>> {
        &self.colors
    }

    pub fn color(&self, x: u32, y: u32) -> Rgb<u8> {
        *self.colors.get_pixel(x, y)
    }

    pub fn get_color(&self, x: i32, y: i32) -> Option<Rgb<u8>> {
        if x < 0 || y < 0 {
            None
        } else {
            self.colors.get_pixel_checked(x as u32, y as u32).copied()
        }
    }

    pub fn nets(&self) -> &[u32] {
        &self.image_nets
    }

    pub fn width(&self) -> u32 {
        self.colors.width()
    }

    pub fn height(&self) -> u32 {
        self.colors.height()
    }

    pub fn power_color(&self) -> Rgb<u8> {
        self.power_color
    }
    pub fn active_gate_color(&self) -> Rgb<u8> {
        self.active_gate_color
    }

    pub fn passive_gate_color(&self) -> Rgb<u8> {
        self.passive_gate_color
    }

    fn net_at(&self, x: u32, y: u32) -> u32 {
        let index = x as usize + y as usize * self.width() as usize;
        self.image_nets[index]
    }

    pub fn pixel(&self, x: u32, y: u32) -> Pixel {
        let Some(&color) = self.colors.get_pixel_checked(x, y) else {
            return Pixel::Insulator;
        };
        let index = x as usize + y as usize * self.width() as usize;

        if hsv_value(color) < Self::INSULATOR_COLOR_MAX_VALUE {
            Pixel::Insulator
        } else if self.active_gate_color == color {
            Pixel::Gate {
                ty: GateType::Active,
                net: self.image_nets[index],
            }
        } else if self.passive_gate_color == color {
            Pixel::Gate {
                ty: GateType::Passive,
                net: self.image_nets[index],
            }
        } else if self.power_color == color {
            Pixel::Power
        } else {
            let index = x as usize + y as usize * self.width() as usize;
            Pixel::Wire {
                color,
                net: self.image_nets[index],
            }
        }
    }

    pub fn get_arrows(&self, x: u32, y: u32) -> (u32, u32) {
        let width = self.width() as usize;
        self.arrows[x as usize + y as usize * width]
    }

    fn get_arrows_mut(&mut self, x: u32, y: u32) -> &mut (u32, u32) {
        let width = self.width() as usize;
        &mut self.arrows[x as usize + y as usize * width]
    }

    fn neighbours_4(&self, x: u32, y: u32) -> [Option<(u32, u32)>; 4] {
        let (w, h) = (self.width(), self.height());
        [
            if y > 0 { Some((x, y - 1)) } else { None },     // Up
            if y + 1 < h { Some((x, y + 1)) } else { None }, // Down
            if x > 0 { Some((x - 1, y)) } else { None },     // Left
            if x + 1 < w { Some((x + 1, y)) } else { None }, // Right
        ]
    }

    pub fn new(colors: ImageBuffer<Rgb<u8>, Vec<u8>>) -> Self {
        let (power_color, active_gate_color, passive_gate_color) =
            CircuitImage::choose_matching_colors(colors.pixels().copied());
        let (width, height) = (colors.width(), colors.height());

        let mut circuit = CircuitImage {
            colors,
            image_nets: vec![NET_OFF; width as usize * height as usize],
            arrows: vec![(0, 0); width as usize * height as usize],

            inputs: Vec::new(),
            outputs: Vec::new(),

            power_color,
            active_gate_color,
            passive_gate_color,

            gates: Vec::new(),
            wires: Vec::new(),

            buffer_gates: Vec::new(),
            not_gates: Vec::new(),
            non_trivial_gates: Vec::new(),
            wires_non_trivial: Vec::new(),
        };

        let mut gates: DenseMap<Gate> = DenseMap::default();
        let mut dfs_stack = Vec::new();

        let mut net_aliases = UnionFind::new(2); // OFF and ON nets

        // arrow_end: (x, y), arrow_direction: (dx, dy)
        let mut arrows_tips: Vec<((u32, u32), (i32, i32))> = Vec::new();

        let compute_arrows = |circuit: &mut CircuitImage, x: u32, y: u32| {
            let Pixel::Wire { color, .. } = circuit.pixel(x, y) else {
                return;
            };

            if circuit.get_arrows(x, y).0 == 0 {
                // Compute Horizontal Arrow
                let mut min_x = x;
                let mut max_x = x;

                while min_x > 0 && circuit.color(min_x - 1, y) == color {
                    min_x -= 1;
                }

                while max_x + 1 < width && circuit.color(max_x + 1, y) == color {
                    max_x += 1;
                }

                for arrow_x in min_x..=max_x {
                    circuit.get_arrows_mut(arrow_x, y).0 = max_x - min_x + 1;
                }
            }

            if circuit.get_arrows(x, y).1 == 0 {
                // Compute Vertical Arrow
                let mut min_y = y;
                let mut max_y = y;

                while min_y > 0 && circuit.color(x, min_y - 1) == color {
                    min_y -= 1;
                }

                while max_y + 1 < height && circuit.color(x, max_y + 1) == color {
                    max_y += 1;
                }

                for arrow_y in min_y..=max_y {
                    circuit.get_arrows_mut(x, arrow_y).1 = max_y - min_y + 1;
                }
            }
        };

        // --- 1. Detect nets ---
        for y in 0..height {
            for x in 0..width {
                let current_pixel = circuit.pixel(x, y);
                match current_pixel {
                    Pixel::Insulator => continue,
                    Pixel::Power => {
                        circuit.set_net(x, y, NET_ON);
                        continue;
                    }
                    // If Net already assigned => skip pixel
                    Pixel::Wire { net, .. } | Pixel::Gate { net, .. } if net != NET_OFF => continue,
                    // Scan wires and gates
                    Pixel::Wire { .. } | Pixel::Gate { .. } => {}
                };

                // Crate net
                let current_net = net_aliases.extend(1);

                // Crate gate
                if let Pixel::Gate { ty, .. } = current_pixel {
                    let mut gate = Gate::new(ty);
                    circuit.set_net(x, y, current_net);
                    dfs_stack.push((x, y));

                    while let Some((x, y)) = dfs_stack.pop() {
                        for (nx, ny) in circuit.neighbours_4(x, y).into_iter().flatten() {
                            match circuit.pixel(nx, ny) {
                                n if n == current_pixel => {}
                                Pixel::Wire { net, .. } => {
                                    if net != NET_OFF && !gate.wires.contains(&net) {
                                        gate.wires.push(net);
                                    }
                                    continue;
                                }
                                Pixel::Gate { net: other_net, .. } => {
                                    if other_net != current_net && other_net != NET_OFF {
                                        if !gate.controls.contains(&other_net) {
                                            gate.controls.push(other_net);
                                        }
                                        if let Some(other_gate) = gates.get_mut(other_net)
                                            && !other_gate.controls.contains(&current_net)
                                        {
                                            other_gate.controls.push(current_net);
                                        }
                                    }
                                    continue;
                                }
                                Pixel::Power => {
                                    if !gate.wires.contains(&NET_ON) {
                                        gate.wires.push(NET_ON);
                                    }
                                    continue;
                                }
                                _ => continue,
                            }

                            circuit.set_net(nx, ny, current_net);
                            dfs_stack.push((nx, ny));
                        }
                    }
                    gates.insert(current_net as usize, gate);
                    continue;
                }

                // First dfs iteration to compute the arrows (setting current_net as visited)
                circuit.set_net(x, y, current_net);
                dfs_stack.push((x, y));

                while let Some((x, y)) = dfs_stack.pop() {
                    for (nx, ny) in circuit.neighbours_4(x, y).into_iter().flatten() {
                        match circuit.pixel(nx, ny) {
                            Pixel::Insulator => continue,
                            Pixel::Power => {
                                if let Pixel::Wire { .. } = current_pixel {
                                    net_aliases.alias(current_net, NET_ON);
                                }
                                continue;
                            }
                            Pixel::Gate { net: gate_net, .. } => {
                                if let Some(gate) = gates.get_mut(gate_net)
                                    && !gate.wires.contains(&current_net)
                                {
                                    gate.wires.push(current_net);
                                }
                                continue;
                            }
                            pixel if pixel != current_pixel => continue,
                            _ => {}
                        }

                        circuit.set_net(nx, ny, current_net);
                        dfs_stack.push((nx, ny));
                    }

                    compute_arrows(&mut circuit, x, y);
                }

                // Second dfs iteration to keep better arrow (setting temp_net as visited)
                let temp_net = current_net + 1;
                circuit.set_net(x, y, temp_net);
                dfs_stack.push((x, y));

                while let Some((x, y)) = dfs_stack.pop() {
                    for (nx, ny) in circuit.neighbours_4(x, y).into_iter().flatten() {
                        if circuit.net_at(nx, ny) == current_net {
                            circuit.set_net(nx, ny, temp_net);
                            dfs_stack.push((nx, ny));
                        }
                    }

                    let (len_x, len_y) = circuit.get_arrows_mut(x, y);
                    if len_x < len_y {
                        *len_x = 0;
                    } else if len_x > len_y {
                        *len_y = 0;
                    } else {
                        *len_x = 0;
                        *len_y = 0;
                    }
                }

                // Third dfs iteration propagate good arrows
                circuit.set_net(x, y, current_net);
                dfs_stack.push((x, y));
                let color = circuit.color(x, y);

                while let Some((x, y)) = dfs_stack.pop() {
                    let mut visit_dfs_neighbour = |circuit: &mut CircuitImage, nx: u32, ny: u32| {
                        if circuit.net_at(nx, ny) == temp_net {
                            circuit.set_net(nx, ny, current_net);
                            dfs_stack.push((nx, ny));
                        }
                    };

                    {
                        // Check Horizontal Arrow
                        let mut min_x = x;
                        let mut max_x = x;

                        while min_x > 0 && circuit.color(min_x - 1, y) == color {
                            min_x -= 1;
                        }

                        while max_x + 1 < width && circuit.color(max_x + 1, y) == color {
                            max_x += 1;
                        }

                        let mut arrow_len = 0;
                        for arrow_x in min_x..=max_x {
                            let (len_x, _) = circuit.get_arrows(arrow_x, y);
                            arrow_len = arrow_len.max(len_x);

                            if arrow_x != x {
                                circuit.set_net(arrow_x, y, current_net);
                                if y + 1 < height {
                                    visit_dfs_neighbour(&mut circuit, arrow_x, y + 1);
                                }
                                if y > 0 {
                                    visit_dfs_neighbour(&mut circuit, arrow_x, y - 1);
                                }
                            }
                        }

                        // Keep arrow
                        if arrow_len > 0 {
                            for arrow_x in min_x..=max_x {
                                circuit.get_arrows_mut(arrow_x, y).0 = arrow_len;
                            }
                        }
                        if min_x > 0 && circuit.pixel(min_x - 1, y).is_wire() {
                            arrows_tips.push(((min_x, y), (-1, 0)));
                        }

                        if max_x + 1 < width && circuit.pixel(max_x + 1, y).is_wire() {
                            arrows_tips.push(((max_x, y), (1, 0)));
                        }
                    }
                    {
                        // Check Vertical Arrow
                        let mut min_y = y;
                        let mut max_y = y;

                        while min_y > 0 && circuit.color(x, min_y - 1) == color {
                            min_y -= 1;
                        }

                        while max_y + 1 < height && circuit.color(x, max_y + 1) == color {
                            max_y += 1;
                        }

                        let mut arrow_len = 0;
                        for arrow_y in min_y..=max_y {
                            let (_, len_y) = circuit.get_arrows(x, arrow_y);
                            arrow_len = arrow_len.max(len_y);

                            if arrow_y != y {
                                circuit.set_net(x, arrow_y, current_net);
                                if x + 1 < width {
                                    visit_dfs_neighbour(&mut circuit, x + 1, arrow_y);
                                }
                                if x > 0 {
                                    visit_dfs_neighbour(&mut circuit, x - 1, arrow_y);
                                }
                            }
                        }

                        // Keep arrow
                        if arrow_len > 0 {
                            for arrow_y in min_y..=max_y {
                                circuit.get_arrows_mut(x, arrow_y).1 = arrow_len;
                            }
                        }

                        if min_y > 0 && circuit.pixel(x, min_y - 1).is_wire() {
                            arrows_tips.push(((x, min_y), (0, -1)));
                        }

                        if max_y + 1 < height && circuit.pixel(x, max_y + 1).is_wire() {
                            arrows_tips.push(((x, max_y), (0, 1)));
                        }
                    }
                }
            }
        }

        for ((x, y), (dx, dy)) in arrows_tips {
            let (arrow_x, arrow_y) = circuit.get_arrows(x, y);
            if dx != 0 && arrow_y != 0 {
                continue;
            }
            if dy != 0 && arrow_x != 0 {
                continue;
            }

            let arrow_net = circuit.net_at(x, y);
            let arrow_color = circuit.color(x, y);

            let (mut x, mut y) = (x as i32, y as i32);
            (x, y) = (x + dx, y + dy);
            let bridge_color = circuit.color(x as u32, y as u32);

            while Some(bridge_color) == circuit.get_color(x, y) {
                (x, y) = (x + dx, y + dy);
            }

            if Some(arrow_color) == circuit.get_color(x, y) {
                net_aliases.alias(arrow_net, circuit.net_at(x as u32, y as u32));
            }
        }

        // let mut bridge_start_coords = Vec::with_capacity(INIT_CAPACITY);
        //
        // // Pixels with all:
        // // - net == bridge
        // // - touching current net (4way)
        // // - pixels touch each other (8way)
        // let mut border_pixels = Vec::with_capacity(INIT_CAPACITY);
        //
        // // Stores the starting points of surrounding pixels. They need all:
        // // - net != current && net != bridge
        // // - touching a border net (8way) && a current net (4way)
        // let mut surround_start_pixels = Vec::with_capacity(INIT_CAPACITY);
        //
        // let mut border_visited = HashSet::with_capacity(INIT_CAPACITY);
        // let mut surround_visited = HashSet::with_capacity(INIT_CAPACITY);
        //
        // // Alternative: Use two UnionFind
        // let mut bridges = HashSet::with_capacity(INIT_CAPACITY);
        //
        // // Try Optimize: currently dfs is storing all nodes like how bfs would do it
        // let mut dfs_stack = Vec::with_capacity(INIT_CAPACITY);
        //
        // // --- Create lables by color & power ---
        //
        // let neighbours_4 = |x: u32, y: u32| {
        //     [
        //         if y > 0 { Some((x, y - 1)) } else { None },      // Up
        //         if y < height { Some((x, y + 1)) } else { None }, // Down
        //         if x > 0 { Some((x - 1, y)) } else { None },      // Left
        //         if x < width { Some((x + 1, y)) } else { None },  // Right
        //     ]
        // };
        //
        // for y in 0..height {
        //     for x in 0..width {
        //         let current_pixel = circuit.pixel(x, y);
        //
        //         let current_net = match current_pixel {
        //             Pixel::Insulator => continue,
        //             Pixel::Power => {
        //                 circuit.set_net(x, y, NET_ON);
        //                 continue;
        //             }
        //             Pixel::Gate { net, .. } => net,
        //             Pixel::Wire { net, .. } => net,
        //         };
        //
        //         if current_net != NET_UNVISITED {
        //             continue;
        //         }
        //
        //         // Start New Net
        //         let current_net = net_aliases.extend(1);
        //         if let Pixel::Gate { ty, .. } = current_pixel {
        //             // "gates" is a non-sparse map => we need also to fill in the
        //             // empty slots with some placeholders.
        //             gates.resize(current_net as usize, None);
        //             gates.push(Some(Gate::new(ty)));
        //         }
        //
        //         circuit.set_net(x, y, current_net);
        //         dfs_stack.push((x, y));
        //
        //         while let Some((x, y)) = dfs_stack.pop() {
        //             for (nx, ny) in neighbours_4(x, y).into_iter().flatten() {
        //                 let neighbour_pixel = circuit.pixel(nx, ny);
        //                 match neighbour_pixel {
        //                     Pixel::Insulator => continue,
        //                     Pixel::Power => {
        //                         if let Some(Some(gate)) = gates.get_mut(current_net as usize) {
        //                             if !gate.wires.contains(&NET_ON) {
        //                                 gate.wires.push(NET_ON);
        //                             }
        //                         } else {
        //                             net_aliases.alias(current_net, NET_ON);
        //                         }
        //                         continue;
        //                     }
        //                     Pixel::Gate {
        //                         ty: neighbour_ty,
        //                         net: neighbour_net,
        //                     } => {
        //                         // Skip if already visited
        //                         if current_net == neighbour_net {
        //                             continue;
        //                         }
        //
        //                         // Skip if current is wire
        //                         let Some(current_ty) = current_pixel.gate_type() else {
        //                             // Register wire (current) touching gate (neighbour)
        //                             if neighbour_net != NET_UNVISITED
        //                                 && let Some(Some(gate)) =
        //                                     gates.get_mut(neighbour_net as usize)
        //                                 && !gate.wires.contains(&current_net)
        //                             {
        //                                 gate.wires.push(current_net);
        //                             }
        //                             continue;
        //                         };
        //
        //                         // Skip if not same gate
        //                         if current_ty != neighbour_ty {
        //                             // Register gate touching opposite gate
        //                             if let Ok([Some(current_gate), Some(neighbour_gate)]) = gates
        //                                 .get_disjoint_mut([
        //                                     current_net as usize,
        //                                     neighbour_net as usize,
        //                                 ])
        //                             {
        //                                 if !current_gate.controls.contains(&neighbour_net) {
        //                                     current_gate.controls.push(neighbour_net);
        //                                 }
        //                                 if !neighbour_gate.controls.contains(&current_net) {
        //                                     neighbour_gate.controls.push(current_net);
        //                                 }
        //                             }
        //                             continue;
        //                         }
        //                     }
        //                     Pixel::Wire {
        //                         color: neighbour_color,
        //                         net: neighbour_net,
        //                     } => {
        //                         // Skip if already visited
        //                         if current_net == neighbour_net {
        //                             continue;
        //                         }
        //
        //                         // Skip if current is gate
        //                         let Some(current_color) = current_pixel.wire_color() else {
        //                             // Register gate (current) touching wire (neighbour)
        //                             if neighbour_net != NET_UNVISITED
        //                                 && let Some(Some(gate)) =
        //                                     gates.get_mut(current_net as usize)
        //                                 && !gate.wires.contains(&neighbour_net)
        //                             {
        //                                 gate.wires.push(neighbour_net);
        //                             }
        //                             continue;
        //                         };
        //
        //                         // Skip if not same wire
        //                         if current_color != neighbour_color {
        //                             // If touching another wire, store it to check bridges later on.
        //                             bridge_start_coords.push(((x, y), (nx, ny)));
        //                             continue;
        //                         }
        //                     }
        //                 }
        //
        //                 // Neighbour is connected and not visited => check it recursively
        //                 circuit.set_net(nx, ny, current_net);
        //                 dfs_stack.push((nx, ny));
        //             }
        //         }
        //     }
        // }
        //
        // // --- Solve wire bridges ---
        //
        // let neighbours_8 = |x: u32, y: u32| {
        //     let (x0, x1) = (0 < x, x + 1 < width);
        //     let (y0, y1) = (0 < y, y + 1 < height);
        //
        //     [
        //         if x1 && y1 { Some((x + 1, y + 1)) } else { None },
        //         if x1 && y0 { Some((x + 1, y - 1)) } else { None },
        //         if x0 && y1 { Some((x - 1, y + 1)) } else { None },
        //         if x0 && y0 { Some((x - 1, y - 1)) } else { None },
        //         if y0 { Some((x, y - 1)) } else { None }, // Up
        //         if y1 { Some((x, y + 1)) } else { None }, // Down
        //         if x0 { Some((x - 1, y)) } else { None }, // Left
        //         if x1 { Some((x + 1, y)) } else { None }, // Right
        //     ]
        // };
        //
        // let mut previous_net = 0;
        //
        // // Loop for each border connecting current net with bridge net
        // for ((first_x, first_y), (first_nx, first_ny)) in bridge_start_coords {
        //     let current_pixel = circuit.pixel(first_x, first_y);
        //     let Pixel::Wire {
        //         color: current_color,
        //         net: current_net,
        //     } = current_pixel
        //     else {
        //         continue;
        //     };
        //
        //     let bridge_pixel = circuit.pixel(first_nx, first_ny);
        //     let Pixel::Wire {
        //         color: _bridge_color,
        //         net: bridge_net,
        //     } = bridge_pixel
        //     else {
        //         continue;
        //     };
        //
        //     if current_net != previous_net {
        //         previous_net = current_net;
        //         border_visited.clear();
        //     }
        //
        //     if border_visited.contains(&(first_nx, first_ny)) {
        //         continue;
        //     }
        //
        //     let (mut normal_x, mut normal_y) = (0., 0.);
        //     border_pixels.clear();
        //     surround_start_pixels.clear();
        //     surround_visited.clear();
        //
        //     // println!(" [NET {current_net}]> Check Border start: {first_nx}, {first_ny}");
        //
        //     let mut dfs_visit = |x: u32, y: u32| {
        //         if border_visited.contains(&(x, y)) {
        //             return false;
        //         }
        //
        //         let net = circuit.net_at(x, y);
        //
        //         if net != bridge_net {
        //             // If it's surround pixel => Store it to check it later on
        //             if net != current_net
        //                 && neighbours_4(x, y)
        //                     .into_iter()
        //                     .flatten()
        //                     .any(|(nx, ny)| circuit.net_at(nx, ny) == current_net)
        //             {
        //                 surround_start_pixels.push((x, y));
        //             }
        //             return false;
        //         }
        //
        //         border_visited.insert((x, y));
        //
        //         // Check if it's border pixel
        //         let (mut pixel_normal_x, mut pixel_normal_y) = (0_f32, 0_f32);
        //         let mut count = 0.;
        //         for (nx, ny) in neighbours_4(x, y).into_iter().flatten() {
        //             if circuit.net_at(nx, ny) == current_net {
        //                 pixel_normal_x += (x as i64 - nx as i64) as f32;
        //                 pixel_normal_y += (y as i64 - ny as i64) as f32;
        //                 count += 1.;
        //             }
        //         }
        //         if count == 0. {
        //             false
        //         } else {
        //             normal_x += pixel_normal_x / count;
        //             normal_y += pixel_normal_y / count;
        //             true
        //         }
        //     };
        //
        //     if !dfs_visit(first_nx, first_ny) {
        //         continue;
        //     }
        //     dfs_stack.push((first_nx, first_ny));
        //
        //     // 1. Find border pixels
        //     while let Some((x, y)) = dfs_stack.pop() {
        //         border_pixels.push((x, y));
        //
        //         for (nx, ny) in neighbours_8(x, y).into_iter().flatten() {
        //             if dfs_visit(nx, ny) {
        //                 dfs_stack.push((nx, ny));
        //             }
        //         }
        //     }
        //
        //     // Intersection limits
        //     let min_normal = border_pixels.len() / 4;
        //     let max_bridge_length = 64 * border_pixels.len();
        //
        //     // Discard if normal is too small (ensures |normal| > 0)
        //     if normal_x.abs() + normal_y.abs() <= min_normal as f32 {
        //         continue;
        //     }
        //
        //     if normal_x.abs() < normal_y.abs() {
        //         normal_x /= normal_y.abs();
        //         normal_y = normal_y.signum();
        //     } else if normal_y.abs() < normal_x.abs() {
        //         normal_y /= normal_x.abs();
        //         normal_x = normal_x.signum();
        //     } else {
        //         normal_x = normal_x.signum();
        //         normal_y = normal_y.signum();
        //     }
        //
        //     // println!(
        //     //     "  [NET {current_net}]> Check Border start: {}, {}, normal: {}, {}",
        //     //     first_nx, first_ny, normal_x, normal_y
        //     // );
        //
        //     // 2. Check angle in surroundings vs bridge
        //     let max_surround_len = 1 + border_pixels.len() / 3;
        //
        //     /// # Minimum Pointyness Parameter
        //     /// Maximum dot product value to accept the bridge.
        //     /// At least one edge of a wire must have this pointyness for it to corss over.
        //     /// range: 0..1
        //     /// - 0: must be at least 90deg
        //     /// - 1: Anything is valid
        //     const MAX_DOT_PRODUCT: f32 = 0.7;
        //
        //     /// # Maximum Pointyness Parameter
        //     /// Minimum dot product value to reject the bridge.
        //     /// No edge of a wire can have this pointyness for it to corss over.
        //     /// range: 0..1
        //     /// - 0: Enforce >=90deg of all corners for bridge.
        //     /// - 1: Nothing is valid (enforce 0deg)
        //     const MIN_DOT_PRODUCT: f32 = 0.95;
        //
        //     let mut surrounding_success = false;
        //
        //     let normal_sq_len = normal_x * normal_x + normal_y * normal_y;
        //
        //     while let Some((x, y)) = surround_start_pixels.pop() {
        //         if surround_visited.contains(&(x, y)) {
        //             continue;
        //         }
        //         surround_visited.insert((x, y));
        //
        //         let surround_net = circuit.net_at(x, y);
        //
        //         let (mut surround_normal_x, mut surround_normal_y) = (0., 0.);
        //
        //         let mut dfs_visit = |x: u32, y: u32| {
        //             // Check if neighbour is surrounding, meaning that:
        //             // - touches other surrouding pixel (8way) (given since parent is surrounding)
        //             // - == parent net
        //             // - touches current_net (4way)
        //
        //             if circuit.net_at(x, y) != surround_net {
        //                 return false;
        //             }
        //
        //             let (mut pixel_normal_x, mut pixel_normal_y) = (0_f32, 0_f32);
        //             let mut count = 0.;
        //
        //             // Check touches current_net & compute slop with it
        //             for (nx, ny) in neighbours_4(x, y).into_iter().flatten() {
        //                 if circuit.net_at(nx, ny) == current_net {
        //                     pixel_normal_x += (x as i64 - nx as i64) as f32;
        //                     pixel_normal_y += (y as i64 - ny as i64) as f32;
        //                     count += 1.;
        //                 }
        //             }
        //
        //             if count == 0. {
        //                 false
        //             } else {
        //                 surround_normal_x += pixel_normal_x / count;
        //                 surround_normal_y += pixel_normal_y / count;
        //                 true
        //             }
        //         };
        //
        //         if !dfs_visit(x, y) {
        //             continue;
        //         }
        //         let mut surround_len = 1;
        //         if surround_len < max_surround_len {
        //             dfs_stack.push((x, y));
        //
        //             // Scan surrounding pixels while computing normal
        //             while let Some((x, y)) = dfs_stack.pop() {
        //                 for (nx, ny) in neighbours_8(x, y).into_iter().flatten() {
        //                     if dfs_visit(nx, ny) {
        //                         surround_len += 1;
        //                         if max_surround_len <= surround_len {
        //                             dfs_stack.clear();
        //                             break;
        //                         }
        //                         dfs_stack.push((nx, ny));
        //                     }
        //                 }
        //             }
        //         }
        //
        //         // Discard if normal too small
        //         let min_surround_normal = surround_len / 6;
        //         if surround_normal_x.abs() + surround_normal_y.abs() < min_surround_normal as f32 {
        //             continue;
        //         }
        //
        //         // Check angle of surrounding pixels vs bridgeed with dot product
        //         let dot = normal_x * surround_normal_x + normal_y * surround_normal_y;
        //         let surround_normal_sq_len =
        //             surround_normal_x * surround_normal_x + surround_normal_y * surround_normal_y;
        //         let dot = dot / (surround_normal_sq_len * normal_sq_len).sqrt();
        //
        //         if MIN_DOT_PRODUCT <= dot {
        //             // Cancel bridge
        //             surrounding_success = false;
        //             break;
        //         }
        //
        //         // Is pointy enough?
        //         if dot <= MAX_DOT_PRODUCT {
        //             surrounding_success = true;
        //         }
        //     }
        //
        //     if !surrounding_success {
        //         continue;
        //     }
        //
        //     // 3. Advance border points until exiting the bridgeed wire.
        //     // Each border pixel is "ray traced" with the normal direction.
        //     while let Some((x, y)) = border_pixels.pop() {
        //         let (mut pre_x, mut pre_y) = (x, y);
        //
        //         for i in 1..=max_bridge_length {
        //             let rx = x as i64 + (normal_x * i as f32) as i64;
        //             let ry = y as i64 + (normal_y * i as f32) as i64;
        //
        //             if rx < 0 || width as i64 <= rx || ry < 0 || height as i64 <= ry {
        //                 break;
        //             }
        //
        //             let (rx, ry) = (rx as u32, ry as u32);
        //
        //             // If we are going in a diagonal, check connection before continuing
        //             let mut ray_ended = false;
        //
        //             let mut ray_hit = |x: u32, y: u32| {
        //                 let Pixel::Wire { color, net } = circuit.pixel(x, y) else {
        //                     ray_ended = true;
        //                     return;
        //                 };
        //
        //                 if net != bridge_net {
        //                     ray_ended = true;
        //                 }
        //
        //                 // If hit, connect wires "current_net" and "net"
        //                 if color == current_color && net != current_net {
        //                     let is_new_alias = bridges.insert((current_net, net));
        //
        //                     // If alias both ways, connect wires
        //                     if is_new_alias && bridges.contains(&(net, current_net)) {
        //                         net_aliases.alias(current_net, net);
        //                     }
        //                 }
        //             };
        //
        //             if rx != pre_x && ry != pre_y {
        //                 ray_hit(rx, pre_y);
        //                 ray_hit(pre_x, ry);
        //             }
        //             (pre_x, pre_y) = (rx, ry);
        //
        //             ray_hit(rx, ry);
        //
        //             if ray_ended {
        //                 break;
        //             }
        //         }
        //     }
        // }

        // // --- Remove unconnected wires and gates ---
        // // Remove permanent gates without controls
        // for (gate_net, gate_slot) in gates.iter_slots() {
        //     if let Some(gate) = gate_slot
        //         && gate.controls.is_empty()
        //     {
        //         if gate.ty == GateType::Passive {
        //             if let Some(&wire_net) = gate.wires.first() {
        //                 for &net in &gate.wires[1..] {
        //                     net_aliases.alias(wire_net, net);
        //                 }
        //                 net_aliases.alias(gate_net as u32, wire_net);
        //             } else {
        //                 net_aliases.alias(gate_net as u32, NET_OFF);
        //             }
        //         }
        //         *gate_slot = None;
        //     }
        // }

        // // Check which wires are connected to which nets
        // let mut used_wires = vec![false; net_aliases.len() as usize];
        // for (_, gate) in gates.iter() {
        //     for &net in &gate.wires {
        //         used_wires[net_aliases.root(net) as usize] = true;
        //     }
        // }
        //
        // let mut gate_count: u32 = 0;
        // for net in 1..net_aliases.len() {
        //     if let Some(gate) = gates.get_mut(net) {
        //         if gate.controls.is_empty() && gate.wires.is_empty() {
        //             gates.remove(net);
        //         } else {
        //             gate_count += 1;
        //         }
        //     } else if !used_wires[net_aliases.root(net) as usize] {
        //         // Remove unconnected wire
        //         // TODO: Remove wires only as an optimization (not for the circuit image)
        //         // net_aliases.alias(net, NET_OFF);
        //     }
        // }

        // --- Apply net aliases & Collect circuit information into the needed format ---

        let (rename_map, net_count) = {
            let gate_count = gates.iter().count();
            let (mut net_aliases, net_count) = net_aliases.into_compact_rename();

            // Rename nets so that all gates are together at the back
            let mut rename_map: Vec<_> = (0..net_count).collect();
            let mut gate_net = net_count - gate_count as u32;
            for (net, net_alias) in net_aliases.iter_mut().enumerate() {
                let current = rename_map[*net_alias as usize];

                if let Some(gate) = gates.take(net) {
                    rename_map[current as usize] = gate_net;
                    rename_map[gate_net as usize] = current;
                    *net_alias = gate_net;
                    gate_net += 1;
                    circuit.gates.push(gate);
                } else {
                    *net_alias = current;
                }
            }
            (net_aliases, net_count)
        };

        // Map image nets
        for net in &mut circuit.image_nets {
            *net = rename_map[*net as usize];
        }

        // Map gate nets & update wires connections
        let wire_count = net_count - circuit.gates.len() as u32;
        circuit.wires = vec![SmallVec::new(); wire_count as usize];
        circuit.wires_non_trivial = vec![SmallVec::new(); wire_count as usize];

        for (gate_index, gate) in &mut circuit.gates.iter_mut().enumerate() {
            for net in &mut gate.controls {
                *net = rename_map[*net as usize];
            }
            for net in &mut gate.wires {
                *net = rename_map[*net as usize];

                if !circuit.wires[*net as usize].contains(&(gate_index as u32)) {
                    circuit.wires[*net as usize].push(gate_index as u32);
                }
            }

            gate.controls.sort_unstable();
            gate.wires.sort_unstable();

            if gate.wires.is_empty() {
                continue;
            }

            debug_assert!(
                gate.wires[0] != NET_OFF,
                "Gate must have wires and never the permanent OFF one"
            );

            if gate.wires[0] == NET_ON {
                // Case gate is trivial
                let trivial_gate = GateConnections {
                    controls: gate.controls.clone(),
                    wires: gate.wires[1..].iter().copied().collect(),
                };

                match gate.ty {
                    GateType::Passive => circuit.buffer_gates.push(trivial_gate),
                    GateType::Active => circuit.not_gates.push(trivial_gate),
                }
            } else {
                // Case gate is not trivial
                let non_tivial_idx = circuit.non_trivial_gates.len() as u32;
                circuit.non_trivial_gates.push(gate.clone());
                for &wire in &gate.wires {
                    circuit.wires_non_trivial[wire as usize].push(non_tivial_idx);
                }
            }
        }

        // Get Inputs (Scanning left pixels)
        let mut prev_input = None;
        for y in 0..height {
            if let Pixel::Wire { net, .. } = circuit.pixel(0, y) {
                if prev_input != Some(net) {
                    prev_input = Some(net);
                    circuit.inputs.push(net);
                }
            } else {
                prev_input = None;
            }
        }

        // Get Outputs (Scanning right pixels)
        let mut prev_output = None;
        for y in 0..height {
            if let Pixel::Wire { net, .. } = circuit.pixel(width - 1, y) {
                if prev_output != Some(net) {
                    prev_output = Some(net);
                    circuit.outputs.push(net);
                }
            } else {
                prev_output = None;
            }
        }

        circuit
    }

    pub fn inputs(&self) -> &[u32] {
        &self.inputs
    }

    pub fn outputs(&self) -> &[u32] {
        &self.outputs
    }

    pub fn gate_count(&self) -> u32 {
        self.gates.len() as u32
    }

    /// Count of nets that are not a gate.
    /// Meaning that it's the number of wires including the two permanent NET_OFF/NET_ON.
    ///
    /// There are allways at least 2 wires (the NET_OFF and NET_ON) even if not in use.
    pub fn wire_count(&self) -> u32 {
        self.wires.len() as u32
    }

    pub fn net_count(&self) -> u32 {
        self.wire_count() + self.gate_count()
    }

    pub fn get_gate(&self, net: u32) -> Option<&Gate> {
        if self.wire_count() <= net {
            self.gates.get((net - self.wire_count()) as usize)
        } else {
            None
        }
    }

    /// Returns an iterator of all the gates connected by the specified wire net.
    pub fn connected_gates(&self, wire_net: u32) -> impl Iterator<Item = u32> {
        self.wires
            .get(wire_net as usize)
            .into_iter()
            .flatten()
            .map(|&gates_idx| self.wire_count() + gates_idx)
    }
}

pub struct CircuitState {
    pub image: Arc<CircuitImage>,
    pub nets: CircuitStateNets,
    pub tick: u64,
}

impl CircuitState {
    pub fn new(circuit: Arc<CircuitImage>) -> Self {
        Self {
            tick: 0,
            nets: CircuitStateNets::new(&circuit),
            image: circuit,
        }
    }
}

impl Clone for CircuitState {
    fn clone(&self) -> Self {
        Self {
            image: self.image.clone(),
            nets: self.nets.clone(),
            tick: self.tick,
        }
    }
    fn clone_from(&mut self, other: &Self) {
        if !Arc::ptr_eq(&self.image, &other.image) {
            self.image.clone_from(&other.image);
        }
        self.nets.clone_from(&other.nets);
        self.tick = other.tick;
    }
}
