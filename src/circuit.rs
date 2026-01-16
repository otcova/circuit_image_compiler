use std::collections::HashSet;

use image::{ImageBuffer, Rgb};

pub struct CircuitImage {
    pub image: ImageBuffer<Rgb<u8>, Vec<u8>>,
    pub nets: Vec<u32>,
    pub net_count: u32,
}

pub fn value(pixel: Rgb<u8>) -> f32 {
    let (r, g, b) = (pixel[0], pixel[1], pixel[2]);
    r.max(g).max(b) as f32 / 255.0
}

pub fn saturation(pixel: Rgb<u8>) -> f32 {
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

pub fn is_background(c: Rgb<u8>) -> bool {
    // saturation(c) <= 0.15
    value(c) <= 0.15
}

pub fn is_active(c: Rgb<u8>) -> bool {
    Rgb::<u8>([0, 166, 47]) == c
}

pub fn is_passive(c: Rgb<u8>) -> bool {
    Rgb::<u8>([0, 80, 152]) == c
}

pub fn is_power(c: Rgb<u8>) -> bool {
    Rgb::<u8>([220, 20, 20]) == c
}

impl CircuitImage {
    pub fn new(image: ImageBuffer<Rgb<u8>, Vec<u8>>) -> CircuitImage {
        // const LABEL_BACKGROUND: u32 = 0;
        const LABEL_POWER: u32 = 1;

        let mut intersection_coords = Vec::new();
        let mut net_aliases = UnionFind::new(2); // background & power nets

        // --- Create lables by color & power ---

        let (width, height) = (image.width(), image.height());
        let mut nets = vec![0; width as usize * height as usize];
        let get_index = |x: u32, y: u32| y as usize * width as usize + x as usize;

        let neighbors_4 = |x: u32, y: u32| {
            [
                if y > 0 { Some((x, y - 1)) } else { None },      // Up
                if y < height { Some((x, y + 1)) } else { None }, // Down
                if x > 0 { Some((x - 1, y)) } else { None },      // Left
                if x < width { Some((x + 1, y)) } else { None },  // Right
            ]
        };

        let mut dfs_stack = Vec::with_capacity(u32::min(width, height) as usize);

        for y in 0..height {
            for x in 0..width {
                if nets[get_index(x, y)] != 0 {
                    continue;
                }

                let current_color = *image.get_pixel(x, y);
                if is_background(current_color) {
                    continue;
                }

                // Start New Blob
                let current_net = if is_power(current_color) {
                    LABEL_POWER
                } else {
                    net_aliases.new_net()
                };
                nets[get_index(x, y)] = current_net;
                dfs_stack.push((x, y));

                while let Some((x, y)) = dfs_stack.pop() {
                    let current_color = *image.get_pixel(x, y);
                    // let current_is_gate = is_active(current_color) || is_passive(current_color);
                    let current_is_power = is_power(current_color);

                    for (nx, ny) in neighbors_4(x, y).into_iter().flatten() {
                        let neighbor_color = *image.get_pixel(nx, ny);
                        if is_background(neighbor_color) {
                            continue;
                        }

                        let neighbor_net = nets[get_index(nx, ny)];
                        if current_net == neighbor_net {
                            continue; // Already visited
                        }

                        let is_glued = current_is_power || is_power(neighbor_color);

                        // If neighbor connected, check it recursively
                        if current_color == neighbor_color || is_glued {
                            if neighbor_net == 0 {
                                if is_power(neighbor_color) {
                                    net_aliases.alias(current_net, LABEL_POWER);
                                }

                                nets[get_index(nx, ny)] = current_net;
                                dfs_stack.push((nx, ny));
                            }
                            continue;
                        }

                        // Touching another wire? Store it to check it later on.
                        if !is_glued {
                            intersection_coords.push(((x, y), (nx, ny)));
                        }
                    }
                }
            }
        }

        // --- Solve wire intersections ---
        let mut border_pixels = Vec::new();

        let mut intersections = HashSet::new();

        let mut border_visited = vec![0_u32; width as usize * height as usize];

        let neighbors_8 = |x: u32, y: u32| {
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

        for ((first_x, first_y), (first_nx, first_ny)) in intersection_coords {
            let current_net = nets[get_index(first_x, first_y)];
            let intersect_net = nets[get_index(first_nx, first_ny)];

            if border_visited[get_index(first_nx, first_ny)] == current_net {
                continue;
            }

            let current_color = *image.get_pixel(first_x, first_y);
            let intersect_color = *image.get_pixel(first_nx, first_ny);

            let (mut slope_x, mut slope_y) = (0., 0.);

            let border_slope = |x: u32, y: u32| {
                if nets[get_index(x, y)] != intersect_net {
                    return None;
                }
                let (mut slope_x, mut slope_y) = (0_f32, 0_f32);
                let mut count = 0.;
                for (nx, ny) in neighbors_4(x, y)
                    .into_iter()
                    .flatten()
                    .filter(|&(nx, ny)| nets[get_index(nx, ny)] == current_net)
                {
                    slope_x += (x as i64 - nx as i64) as f32;
                    slope_y += (y as i64 - ny as i64) as f32;
                    count += 1.;
                }
                if count == 0. {
                    None
                } else {
                    Some((slope_x / count, slope_y / count))
                }
            };

            if let Some((sx, sy)) = border_slope(first_nx, first_ny) {
                slope_x += sx;
                slope_y += sy;
            } else {
                continue;
            };

            border_visited[get_index(first_nx, first_ny)] = current_net;
            dfs_stack.push((first_nx, first_ny));

            // 1. Find border pixels
            border_pixels.clear();
            while let Some((x, y)) = dfs_stack.pop() {
                border_pixels.push((x, y));

                for (nx, ny) in neighbors_8(x, y).into_iter().flatten() {
                    if border_visited[get_index(nx, ny)] == current_net {
                        continue;
                    }

                    if let Some((sx, sy)) = border_slope(nx, ny) {
                        slope_x += sx;
                        slope_y += sy;

                        border_visited[get_index(nx, ny)] = current_net;
                        dfs_stack.push((nx, ny));
                    }
                }
            }

            // Intersection limits
            let min_slope = border_pixels.len() / 4;
            let max_intersection_length = 64 * border_pixels.len();

            if slope_x.abs() + slope_y.abs() < min_slope as f32 {
                continue;
            }

            if slope_x.abs() < slope_y.abs() {
                slope_x /= slope_y.abs();
                slope_y = slope_y.signum();
            } else if slope_y.abs() < slope_x.abs() {
                slope_y /= slope_x.abs();
                slope_x = slope_x.signum();
            } else if slope_x == 0. && slope_y == 0. {
                // Ignore ambiguous circle-shaped border
                continue;
            } else {
                slope_x = slope_x.signum();
                slope_y = slope_y.signum();
            }

            // 2. Check ~90deg angle in surroundings of intersection
            // TODO

            // 3. Advance border points until exiting the intersected wire
            while let Some((x, y)) = border_pixels.pop() {
                let (mut pre_x, mut pre_y) = (x, y);

                for i in 1..=max_intersection_length {
                    let rx = x as i64 + (slope_x * i as f32) as i64;
                    let ry = y as i64 + (slope_y * i as f32) as i64;

                    if rx < 0 || width as i64 <= rx || ry < 0 || height as i64 <= ry {
                        break;
                    }

                    let (rx, ry) = (rx as u32, ry as u32);

                    // If we are going in a diagonal, check connection before continuing
                    let mut ray_ended = false;

                    let mut ray_hit = |x: u32, y: u32| {
                        let color = *image.get_pixel(x, y);
                        let net = nets[get_index(x, y)];

                        if color != intersect_color {
                            ray_ended = true;
                        }

                        // If hit, connect wires "current_net" and "net"
                        if color == current_color && current_net != net {
                            let is_new_alias = intersections.insert((current_net, net));

                            // If alias both ways, connect wires
                            if is_new_alias && intersections.contains(&(net, current_net)) {
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

        // Merge nets of intersecting wires
        let net_count = net_aliases.compact();
        for net in &mut nets {
            *net = net_aliases.parent(*net);
        }

        CircuitImage {
            image,
            nets,
            net_count: net_count - 1,
        }
    }

    pub fn get_net(&self, x: u32, y: u32) -> u32 {
        self.nets[y as usize * self.image.width() as usize + x as usize]
    }
}

struct UnionFind {
    // Invariant: aliases[i] <= i
    aliases: Vec<u32>,
}

impl UnionFind {
    pub fn new(net_count: u32) -> UnionFind {
        UnionFind {
            aliases: (0..net_count).collect(),
        }
    }

    pub fn new_net(&mut self) -> u32 {
        let i = u32::try_from(self.aliases.len()).unwrap();
        self.aliases.push(i);
        i
    }

    /// Finds the root and compresses the path to it by half
    pub fn root(&mut self, mut i: u32) -> u32 {
        while i != self.aliases[i as usize] {
            let parent = self.aliases[i as usize];
            let grandparent = self.aliases[parent as usize];

            self.aliases[i as usize] = grandparent;
            i = grandparent;
        }
        i
    }

    pub fn parent(&mut self, i: u32) -> u32 {
        self.aliases[i as usize]
    }

    /// Return the root of the resulting alias
    pub fn alias(&mut self, a: u32, b: u32) -> u32 {
        let root_a = self.root(a);
        let root_b = self.root(b);

        if root_a < root_b {
            self.aliases[root_b as usize] = root_a;
            root_a
        } else {
            self.aliases[root_a as usize] = root_b;
            root_b
        }
    }

    /// Resolves dependencies, and removes holes
    /// (meaning that all roots will be contiguous from 0 to root_count)
    ///
    /// Returns the amount of disctinct roots.
    pub fn compact(&mut self) -> u32 {
        let mut root_count = 0;
        for i in 0..self.aliases.len() {
            if self.aliases[i] == i as u32 {
                self.aliases[i] = root_count;
                root_count += 1;
            } else {
                self.aliases[i] = self.aliases[self.aliases[i] as usize];
            }
        }
        root_count
    }

    // /// Resove the backwards dependencies.
    // /// After this call, for all i: parent(i) == root(i)
    // pub fn flatten(&mut self) {
    //     for i in 0..self.aliases.len() {
    //         self.aliases[i] = self.aliases[self.aliases[i] as usize];
    //     }
    // }
}
