use std::collections::HashSet;

use image::{ImageBuffer, Rgb};
use smallvec::SmallVec;

// Permanent unconditional off
const NET_OFF: u32 = 0;

// Permanent unconditional on (power)
const NET_ON: u32 = 1;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum GateType {
    Passive,
    Active,
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

    // Returns None if it's not a gate
    pub fn gate_type(self) -> Option<GateType> {
        if let Pixel::Gate { ty, .. } = self {
            Some(ty)
        } else {
            None
        }
    }

    // Returns None if it's not a wire
    pub fn wire_color(self) -> Option<Rgb<u8>> {
        if let Pixel::Wire { color, .. } = self {
            Some(color)
        } else {
            None
        }
    }
}

#[derive(Clone)]
pub struct Gate {
    pub ty: GateType,
    /// Gate-Gate contacts
    pub inputs: SmallVec<[u32; 4]>,
    /// Gate-Wire contacts
    pub outputs: SmallVec<[u32; 4]>,
}

impl Gate {
    pub fn new(ty: GateType) -> Gate {
        Gate {
            ty,
            inputs: SmallVec::new(),
            outputs: SmallVec::new(),
        }
    }
}

pub struct Circuit {
    pub image: CircuitImage,
    gates: Vec<Option<Gate>>,
}

pub struct CircuitImage {
    colors: ImageBuffer<Rgb<u8>, Vec<u8>>,
    nets: Vec<u32>,
    /// Invariant: There are allways at least 2 nets 0 (allways off) and 1 (allways on)
    /// even if not in use.
    net_count: u32,
}

impl CircuitImage {
    /// Initializes a new circuit image with all pixels set to NET_OFF.
    pub fn new(colors: ImageBuffer<Rgb<u8>, Vec<u8>>) -> CircuitImage {
        CircuitImage {
            nets: vec![NET_OFF; colors.width() as usize * colors.height() as usize],
            net_count: 2,
            colors,
        }
    }

    pub fn set_net(&mut self, x: u32, y: u32, net: u32) {
        let index = x as usize + y as usize * self.width() as usize;
        self.nets[index] = net;
    }

    pub fn colors(&self) -> &ImageBuffer<Rgb<u8>, Vec<u8>> {
        &self.colors
    }

    pub fn nets(&self) -> &[u32] {
        &self.nets
    }

    pub fn net_count(&self) -> u32 {
        self.net_count
    }

    pub fn width(&self) -> u32 {
        self.colors.width()
    }

    pub fn height(&self) -> u32 {
        self.colors.height()
    }

    fn net_at(&self, x: u32, y: u32) -> u32 {
        let index = x as usize + y as usize * self.width() as usize;
        self.nets[index]
    }

    pub fn pixel(&self, x: u32, y: u32) -> Pixel {
        let Some(&color) = self.colors.get_pixel_checked(x, y) else {
            return Pixel::Insulator;
        };
        let index = x as usize + y as usize * self.width() as usize;

        if hsv_value(color) <= 0.15 {
            Pixel::Insulator
        } else if Rgb::<u8>([0, 166, 47]) == color {
            Pixel::Gate {
                ty: GateType::Active,
                net: self.nets[index],
            }
        } else if Rgb::<u8>([0, 80, 152]) == color {
            Pixel::Gate {
                ty: GateType::Passive,
                net: self.nets[index],
            }
        } else if Rgb::<u8>([220, 20, 20]) == color {
            Pixel::Power
        } else {
            let index = x as usize + y as usize * self.width() as usize;
            Pixel::Wire {
                color,
                net: self.nets[index],
            }
        }
    }
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

impl Circuit {
    pub fn new(colors: ImageBuffer<Rgb<u8>, Vec<u8>>) -> Circuit {
        // Try Optimize: Check if this is relevant
        const INIT_CAPACITY: usize = 64;

        let mut image = CircuitImage::new(colors);
        const NET_UNVISITED: u32 = NET_OFF;
        let (width, height) = (image.width(), image.height());

        // A non-sparse map from net (u32) to Gate
        let mut gates = Vec::with_capacity(INIT_CAPACITY);

        let mut net_aliases = UnionFind::new(2); // OFF and ON nets

        let mut bridge_start_coords = Vec::with_capacity(INIT_CAPACITY);

        // Pixels with all:
        // - net == bridge
        // - touching current net (4way)
        // - pixels touch each other (8way)
        let mut border_pixels = Vec::with_capacity(INIT_CAPACITY);

        // Stores the starting points of surrounding pixels. They need all:
        // - net != current && net != bridge
        // - touching a border net (8way) && a current net (4way)
        let mut surround_start_pixels = Vec::with_capacity(INIT_CAPACITY);

        let mut border_visited = HashSet::with_capacity(INIT_CAPACITY);
        let mut surround_visited = HashSet::with_capacity(INIT_CAPACITY);

        // Alternative: Use two UnionFind
        let mut bridges = HashSet::with_capacity(INIT_CAPACITY);

        // Try Optimize: currently dfs is storing all nodes like how bfs would do it
        let mut dfs_stack = Vec::with_capacity(INIT_CAPACITY);

        // --- Create lables by color & power ---

        let neighbours_4 = |x: u32, y: u32| {
            [
                if y > 0 { Some((x, y - 1)) } else { None },      // Up
                if y < height { Some((x, y + 1)) } else { None }, // Down
                if x > 0 { Some((x - 1, y)) } else { None },      // Left
                if x < width { Some((x + 1, y)) } else { None },  // Right
            ]
        };

        for y in 0..height {
            for x in 0..width {
                let current_pixel = image.pixel(x, y);

                let current_net = match current_pixel {
                    Pixel::Insulator => continue,
                    Pixel::Power => {
                        image.set_net(x, y, NET_ON);
                        continue;
                    }
                    Pixel::Gate { net, .. } => net,
                    Pixel::Wire { net, .. } => net,
                };

                if current_net != NET_UNVISITED {
                    continue;
                }

                // Start New Net
                let current_net = net_aliases.extend(1);
                if let Pixel::Gate { ty, .. } = current_pixel {
                    // "gates" is a non-sparse map => we need also to fill in the
                    // empty slots with some placeholders.
                    gates.resize(current_net as usize, None);
                    gates.push(Some(Gate::new(ty)));
                }

                image.set_net(x, y, current_net);
                dfs_stack.push((x, y));

                while let Some((x, y)) = dfs_stack.pop() {
                    for (nx, ny) in neighbours_4(x, y).into_iter().flatten() {
                        let neighbour_pixel = image.pixel(nx, ny);
                        match neighbour_pixel {
                            Pixel::Insulator => continue,
                            Pixel::Power => {
                                if let Some(Some(gate)) = gates.get_mut(current_net as usize) {
                                    if !gate.outputs.contains(&NET_ON) {
                                        gate.outputs.push(NET_ON);
                                    }
                                } else {
                                    net_aliases.alias(current_net, NET_ON);
                                }
                                continue;
                            }
                            Pixel::Gate {
                                ty: neighbour_ty,
                                net: neighbour_net,
                            } => {
                                // Skip if already visited
                                if current_net == neighbour_net {
                                    continue;
                                }

                                // Skip if current is wire
                                let Some(current_ty) = current_pixel.gate_type() else {
                                    // Register wire (current) touching gate (neighbour)
                                    if neighbour_net != NET_UNVISITED
                                        && let Some(Some(gate)) =
                                            gates.get_mut(neighbour_net as usize)
                                        && !gate.outputs.contains(&current_net)
                                    {
                                        gate.outputs.push(current_net);
                                    }
                                    continue;
                                };

                                // Skip if not same gate
                                if current_ty != neighbour_ty {
                                    // Register gate touching opposite gate
                                    if let Ok([Some(current_gate), Some(neighbour_gate)]) = gates
                                        .get_disjoint_mut([
                                            current_net as usize,
                                            neighbour_net as usize,
                                        ])
                                    {
                                        if !current_gate.inputs.contains(&neighbour_net) {
                                            current_gate.inputs.push(neighbour_net);
                                        }
                                        if !neighbour_gate.inputs.contains(&current_net) {
                                            neighbour_gate.inputs.push(current_net);
                                        }
                                    }
                                    continue;
                                }
                            }
                            Pixel::Wire {
                                color: neighbour_color,
                                net: neighbour_net,
                            } => {
                                // Skip if already visited
                                if current_net == neighbour_net {
                                    continue;
                                }

                                // Skip if current is gate
                                let Some(current_color) = current_pixel.wire_color() else {
                                    // Register gate (current) touching wire (neighbour)
                                    if neighbour_net != NET_UNVISITED
                                        && let Some(Some(gate)) =
                                            gates.get_mut(current_net as usize)
                                        && !gate.outputs.contains(&neighbour_net)
                                    {
                                        gate.outputs.push(neighbour_net);
                                    }
                                    continue;
                                };

                                // Skip if not same wire
                                if current_color != neighbour_color {
                                    // If touching another wire, store it to check bridges later on.
                                    bridge_start_coords.push(((x, y), (nx, ny)));
                                    continue;
                                }
                            }
                        }

                        // Neighbour is connected and not visited => check it recursively
                        image.set_net(nx, ny, current_net);
                        dfs_stack.push((nx, ny));
                    }
                }
            }
        }

        // --- Solve wire bridges ---

        let neighbours_8 = |x: u32, y: u32| {
            let (x0, x1) = (0 < x, x + 1 < width);
            let (y0, y1) = (0 < y, y + 1 < height);

            [
                if x1 && y1 { Some((x + 1, y + 1)) } else { None },
                if x1 && y0 { Some((x + 1, y - 1)) } else { None },
                if x0 && y1 { Some((x - 1, y + 1)) } else { None },
                if x0 && y0 { Some((x - 1, y - 1)) } else { None },
                if y0 { Some((x, y - 1)) } else { None }, // Up
                if y1 { Some((x, y + 1)) } else { None }, // Down
                if x0 { Some((x - 1, y)) } else { None }, // Left
                if x1 { Some((x + 1, y)) } else { None }, // Right
            ]
        };

        let mut previous_net = 0;

        // Loop for each border connecting current net with bridge net
        for ((first_x, first_y), (first_nx, first_ny)) in bridge_start_coords {
            let current_pixel = image.pixel(first_x, first_y);
            let Pixel::Wire {
                color: current_color,
                net: current_net,
            } = current_pixel
            else {
                continue;
            };

            let bridge_pixel = image.pixel(first_nx, first_ny);
            let Pixel::Wire {
                color: _bridge_color,
                net: bridge_net,
            } = bridge_pixel
            else {
                continue;
            };

            if current_net != previous_net {
                previous_net = current_net;
                border_visited.clear();
            }

            if border_visited.contains(&(first_nx, first_ny)) {
                continue;
            }

            let (mut normal_x, mut normal_y) = (0., 0.);
            border_pixels.clear();
            surround_start_pixels.clear();
            surround_visited.clear();

            // println!(" [NET {current_net}]> Check Border start: {first_nx}, {first_ny}");

            let mut dfs_visit = |x: u32, y: u32| {
                if border_visited.contains(&(x, y)) {
                    return false;
                }

                let net = image.net_at(x, y);

                if net != bridge_net {
                    // If it's surround pixel => Store it to check it later on
                    if net != current_net
                        && neighbours_4(x, y)
                            .into_iter()
                            .flatten()
                            .any(|(nx, ny)| image.net_at(nx, ny) == current_net)
                    {
                        surround_start_pixels.push((x, y));
                    }
                    return false;
                }

                border_visited.insert((x, y));

                // Check if it's border pixel
                let (mut pixel_normal_x, mut pixel_normal_y) = (0_f32, 0_f32);
                let mut count = 0.;
                for (nx, ny) in neighbours_4(x, y).into_iter().flatten() {
                    if image.net_at(nx, ny) == current_net {
                        pixel_normal_x += (x as i64 - nx as i64) as f32;
                        pixel_normal_y += (y as i64 - ny as i64) as f32;
                        count += 1.;
                    }
                }
                if count == 0. {
                    false
                } else {
                    normal_x += pixel_normal_x / count;
                    normal_y += pixel_normal_y / count;
                    true
                }
            };

            if !dfs_visit(first_nx, first_ny) {
                continue;
            }
            dfs_stack.push((first_nx, first_ny));

            // 1. Find border pixels
            while let Some((x, y)) = dfs_stack.pop() {
                border_pixels.push((x, y));

                for (nx, ny) in neighbours_8(x, y).into_iter().flatten() {
                    if dfs_visit(nx, ny) {
                        dfs_stack.push((nx, ny));
                    }
                }
            }

            // Intersection limits
            let min_normal = border_pixels.len() / 4;
            let max_bridge_length = 64 * border_pixels.len();

            // Discard if normal is too small (ensures |normal| > 0)
            if normal_x.abs() + normal_y.abs() <= min_normal as f32 {
                continue;
            }

            if normal_x.abs() < normal_y.abs() {
                normal_x /= normal_y.abs();
                normal_y = normal_y.signum();
            } else if normal_y.abs() < normal_x.abs() {
                normal_y /= normal_x.abs();
                normal_x = normal_x.signum();
            } else {
                normal_x = normal_x.signum();
                normal_y = normal_y.signum();
            }

            // println!(
            //     "  [NET {current_net}]> Check Border start: {}, {}, normal: {}, {}",
            //     first_nx, first_ny, normal_x, normal_y
            // );

            // 2. Check angle in surroundings vs bridge
            let max_surround_len = 1 + border_pixels.len() / 3;

            /// # Minimum Pointyness Parameter
            /// Maximum dot product value to accept the bridge.
            /// At least one edge of a wire must have this pointyness for it to corss over.
            /// range: 0..1
            /// - 0: must be at least 90deg
            /// - 1: Anything is valid
            const MAX_DOT_PRODUCT: f32 = 0.7;

            /// # Maximum Pointyness Parameter
            /// Minimum dot product value to reject the bridge.
            /// No edge of a wire can have this pointyness for it to corss over.
            /// range: 0..1
            /// - 0: Enforce >=90deg of all corners for bridge.
            /// - 1: Nothing is valid (enforce 0deg)
            const MIN_DOT_PRODUCT: f32 = 0.95;

            let mut surrounding_success = false;

            let normal_sq_len = normal_x * normal_x + normal_y * normal_y;

            while let Some((x, y)) = surround_start_pixels.pop() {
                if surround_visited.contains(&(x, y)) {
                    continue;
                }
                surround_visited.insert((x, y));

                let surround_net = image.net_at(x, y);

                let (mut surround_normal_x, mut surround_normal_y) = (0., 0.);

                let mut dfs_visit = |x: u32, y: u32| {
                    // Check if neighbour is surrounding, meaning that:
                    // - touches other surrouding pixel (8way) (given since parent is surrounding)
                    // - == parent net
                    // - touches current_net (4way)

                    if image.net_at(x, y) != surround_net {
                        return false;
                    }

                    let (mut pixel_normal_x, mut pixel_normal_y) = (0_f32, 0_f32);
                    let mut count = 0.;

                    // Check touches current_net & compute slop with it
                    for (nx, ny) in neighbours_4(x, y).into_iter().flatten() {
                        if image.net_at(nx, ny) == current_net {
                            pixel_normal_x += (x as i64 - nx as i64) as f32;
                            pixel_normal_y += (y as i64 - ny as i64) as f32;
                            count += 1.;
                        }
                    }

                    if count == 0. {
                        false
                    } else {
                        surround_normal_x += pixel_normal_x / count;
                        surround_normal_y += pixel_normal_y / count;
                        true
                    }
                };

                if !dfs_visit(x, y) {
                    continue;
                }
                let mut surround_len = 1;
                if surround_len < max_surround_len {
                    dfs_stack.push((x, y));

                    // Scan surrounding pixels while computing normal
                    while let Some((x, y)) = dfs_stack.pop() {
                        for (nx, ny) in neighbours_8(x, y).into_iter().flatten() {
                            if dfs_visit(nx, ny) {
                                surround_len += 1;
                                if max_surround_len <= surround_len {
                                    dfs_stack.clear();
                                    break;
                                }
                                dfs_stack.push((nx, ny));
                            }
                        }
                    }
                }

                // Discard if normal too small
                let min_surround_normal = surround_len / 6;
                if surround_normal_x.abs() + surround_normal_y.abs() < min_surround_normal as f32 {
                    continue;
                }

                // Check angle of surrounding pixels vs bridgeed with dot product
                let dot = normal_x * surround_normal_x + normal_y * surround_normal_y;
                let surround_normal_sq_len =
                    surround_normal_x * surround_normal_x + surround_normal_y * surround_normal_y;
                let dot = dot / (surround_normal_sq_len * normal_sq_len).sqrt();

                if MIN_DOT_PRODUCT <= dot {
                    // Cancel bridge
                    surrounding_success = false;
                    break;
                }

                // Is pointy enough?
                if dot <= MAX_DOT_PRODUCT {
                    surrounding_success = true;
                }
            }

            if !surrounding_success {
                continue;
            }

            // 3. Advance border points until exiting the bridgeed wire.
            // Each border pixel is "ray traced" with the normal direction.
            while let Some((x, y)) = border_pixels.pop() {
                let (mut pre_x, mut pre_y) = (x, y);

                for i in 1..=max_bridge_length {
                    let rx = x as i64 + (normal_x * i as f32) as i64;
                    let ry = y as i64 + (normal_y * i as f32) as i64;

                    if rx < 0 || width as i64 <= rx || ry < 0 || height as i64 <= ry {
                        break;
                    }

                    let (rx, ry) = (rx as u32, ry as u32);

                    // If we are going in a diagonal, check connection before continuing
                    let mut ray_ended = false;

                    let mut ray_hit = |x: u32, y: u32| {
                        let Pixel::Wire { color, net } = image.pixel(x, y) else {
                            ray_ended = true;
                            return;
                        };

                        if net != bridge_net {
                            ray_ended = true;
                        }

                        // If hit, connect wires "current_net" and "net"
                        if color == current_color && net != current_net {
                            let is_new_alias = bridges.insert((current_net, net));

                            // If alias both ways, connect wires
                            if is_new_alias && bridges.contains(&(net, current_net)) {
                                net_aliases.alias(current_net, net);
                            }
                        }
                    };

                    if rx != pre_x && ry != pre_y {
                        ray_hit(rx, pre_y);
                        ray_hit(pre_x, ry);
                    }
                    (pre_x, pre_y) = (rx, ry);

                    ray_hit(rx, ry);

                    if ray_ended {
                        break;
                    }
                }
            }
        }

        // --- Apply net aliases ---
        let rename_map = {
            let (mut net_aliases, net_count) = net_aliases.into_compact_rename();
            image.net_count = net_count;

            // Rename nets so that all gates are together at the back
            let mut rename_map: Vec<_> = (0..net_count).collect();
            let mut right = net_count;
            for (net, net_alias) in net_aliases.iter_mut().enumerate() {
                let current = rename_map[*net_alias as usize];
                if gates.get(net).is_some_and(|g| g.is_some()) {
                    right -= 1;
                    rename_map[current as usize] = right;
                    rename_map[right as usize] = current;
                    *net_alias = right;
                } else {
                    *net_alias = current;
                }
            }
            net_aliases
        };

        for net in &mut image.nets {
            if gates.get(*net as usize).is_none_or(|g| g.is_none()) {
                *net = rename_map[*net as usize];
            }
        }
        for gate in gates.iter_mut().flatten() {
            for _ in 0..gate.inputs.len() {
                let net = rename_map[gate.inputs.swap_remove(0) as usize];
                if !gate.inputs.contains(&net) {
                    gate.inputs.push(net);
                }
            }
            for _ in 0..gate.outputs.len() {
                let net = rename_map[gate.outputs.swap_remove(0) as usize];
                if !gate.outputs.contains(&net) {
                    gate.outputs.push(net);
                }
            }
        }

        Circuit { image, gates }
    }

    pub fn net_count(&self) -> u32 {
        self.image.net_count()
    }

    pub fn get_gate(&self, net: u32) -> Option<&Gate> {
        self.gates.get(net as usize).and_then(|r| r.as_ref())
    }
}

struct UnionFind {
    // Invariant: parents[i] <= i
    // Invariant: len <= u32::MAX + 1
    parents: Vec<u32>,
}

#[allow(dead_code)]
impl UnionFind {
    pub fn new(node_count: u32) -> UnionFind {
        UnionFind {
            parents: (0..node_count).collect(),
        }
    }

    pub fn from_vec_unchecked(parents: Vec<u32>) -> UnionFind {
        UnionFind { parents }
    }

    /// Crates n consecutive new nodes.
    /// Returns first node created.
    pub fn extend(&mut self, n: u32) -> u32 {
        let first = self.parents.len();

        // Check Invariant "len <= u32::MAX + 1"
        debug_assert!(n <= u32::MAX - (first - 1) as u32);

        let first = first as u32;
        self.parents.extend(first..first + n);
        first
    }

    /// Finds the root and compresses the path to it by half
    pub fn root(&mut self, mut i: u32) -> u32 {
        while i != self.parents[i as usize] {
            let parent = self.parents[i as usize];
            let grandparent = self.parents[parent as usize];

            self.parents[i as usize] = grandparent;
            i = grandparent;
        }
        i
    }

    /// Returns the amount of nodes independently of the aliases
    pub fn len(&self) -> u32 {
        self.parents.len() as u32
    }

    pub fn parent(&mut self, i: u32) -> u32 {
        self.parents[i as usize]
    }

    /// Return the root of the resulting alias
    pub fn alias(&mut self, a: u32, b: u32) -> u32 {
        let root_a = self.root(a);
        let root_b = self.root(b);

        if root_a < root_b {
            self.parents[root_b as usize] = root_a;
            root_a
        } else {
            self.parents[root_a as usize] = root_b;
            root_b
        }
    }

    /// Converts the UnionFind alias-datastructure into a dense rename map.
    ///
    /// - The parents of i (vec[i]) will point to what was the root of the node i (root(i)).
    /// - All elements of the returned vec will be in 0..=(the amount of disctinct roots - 1)
    ///   without any value missing.
    ///
    /// Returns (the Vec rename mapping, the amount of disctinct roots).
    pub fn into_compact_rename(mut self) -> (Vec<u32>, u32) {
        let mut root_count = 0;
        for i in 0..self.parents.len() {
            if self.parents[i] == i as u32 {
                self.parents[i] = root_count;
                root_count += 1;
            } else {
                self.parents[i] = self.parents[self.parents[i] as usize];
            }
        }
        (self.parents, root_count)
    }

    /// Resove the backwards dependencies.
    /// After this call, for all i: parent(i) == root(i)
    pub fn flatten(&mut self) {
        for i in 0..self.parents.len() {
            self.parents[i] = self.parents[self.parents[i] as usize];
        }
    }

    /// Removes all aplied aliases.
    /// Sets the root of every node to itselfs.
    pub fn reset_aliases(&mut self) {
        for i in 0..self.parents.len() {
            self.parents[i] = i as u32;
        }
    }
}
