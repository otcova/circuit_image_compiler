use image::{ImageBuffer, Rgb};

pub struct CircuitImage {
    pub image: ImageBuffer<Rgb<u8>, Vec<u8>>,
    pub labels: Vec<u32>,
    pub label_count: u32,
}

impl CircuitImage {
    pub fn new(image: ImageBuffer<Rgb<u8>, Vec<u8>>) -> CircuitImage {
        let ignore_color = Rgb::<u8>([25, 3, 58]);
        let glue_color = Rgb::<u8>([56, 174, 11]);

        let mut intersection_coords = Vec::new();

        // --- Create lables by color & glue ---

        let (width, height) = (image.width(), image.height());
        let mut labels = vec![0; width as usize * height as usize];
        let get_index = |x: u32, y: u32| y as usize * width as usize + x as usize;

        let neighbors_4 = |x: u32, y: u32| {
            [
                if y > 0 { Some((x, y - 1)) } else { None },      // Up
                if y < height { Some((x, y + 1)) } else { None }, // Down
                if x > 0 { Some((x - 1, y)) } else { None },      // Left
                if x < width { Some((x + 1, y)) } else { None },  // Right
            ]
        };

        let mut current_label: u32 = 0;
        let mut dfs_stack = Vec::with_capacity(u32::min(width, height) as usize);

        for y in 0..height {
            for x in 0..width {
                if labels[get_index(x, y)] != 0 {
                    continue;
                }

                let start_color = *image.get_pixel(x, y);
                if start_color == ignore_color {
                    continue;
                }

                // Start New Blob
                current_label = current_label.strict_add(1);
                labels[get_index(x, y)] = current_label;
                dfs_stack.push((x, y));

                while let Some((x, y)) = dfs_stack.pop() {
                    // We need the color of the current pixel to determine connectivity
                    let current_color = *image.get_pixel(x, y);

                    for (nx, ny) in neighbors_4(x, y).into_iter().flatten() {
                        let neighbor_color = *image.get_pixel(nx, ny);
                        if neighbor_color == ignore_color {
                            continue;
                        }

                        let neighbor_label = labels[get_index(nx, ny)];

                        // touches color || touches glue || is glue?
                        let is_connected = (current_color == neighbor_color)
                            || (current_color == glue_color)
                            || (neighbor_color == glue_color);

                        if !is_connected {
                            // Touching another wire? Store it to check it later on.
                            if neighbor_label != current_label {
                                intersection_coords.push(((x, y), (nx, ny)));
                            }
                            continue;
                        }

                        if neighbor_label == 0 {
                            labels[get_index(nx, ny)] = current_label;
                            dfs_stack.push((nx, ny));
                        }
                    }
                }
            }
        }

        // --- Detect wire intersections ---
        let mut border_line = Vec::new();

        let mut label_mapping: Vec<_> = (0..=current_label).collect();

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
            let current_label = labels[get_index(first_x, first_y)];
            let intersect_label = labels[get_index(first_nx, first_ny)];

            if border_visited[get_index(first_x, first_y)] == current_label {
                continue;
            }

            // TODO: find root
            if label_mapping[current_label as usize] == label_mapping[intersect_label as usize] {
                continue; // Skip intersections connected somewhere else
            }

            let current_color = *image.get_pixel(first_x, first_y);
            let intersect_color = *image.get_pixel(first_nx, first_ny);

            let (mut slope_x, mut slope_y) = (0., 0.);

            let border_slope = |x: u32, y: u32| {
                if *image.get_pixel(x, y) != current_color {
                    return None;
                }
                let (mut slope_x, mut slope_y) = (0_f32, 0_f32);
                let mut count = 0.;
                for (nx, ny) in neighbors_4(x, y)
                    .into_iter()
                    .flatten()
                    .filter(|&(nx, ny)| *image.get_pixel(nx, ny) == intersect_color)
                {
                    slope_x += (nx as i64 - x as i64) as f32;
                    slope_y += (ny as i64 - y as i64) as f32;
                    count += 1.;
                }
                if count == 0. {
                    None
                } else {
                    Some((slope_x / count, slope_y / count))
                }
            };

            let Some((sx, sy)) = border_slope(first_x, first_y) else {
                continue;
            };
            slope_x += sx;
            slope_y += sy;

            border_visited[get_index(first_x, first_y)] = current_label;
            dfs_stack.push((first_x, first_y));

            // 1. Scan border pixels
            border_line.clear();
            while let Some((x, y)) = dfs_stack.pop() {
                border_line.push((x, y));

                for (nx, ny) in neighbors_8(x, y).into_iter().flatten() {
                    if border_visited[get_index(nx, ny)] == current_label {
                        continue;
                    }

                    if let Some((sx, sy)) = border_slope(nx, ny) {
                        slope_x += sx;
                        slope_y += sy;

                        border_visited[get_index(nx, ny)] = current_label;
                        dfs_stack.push((nx, ny));
                    }
                }
            }

            // Intersection limits
            let min_slope = border_line.len() / 4;
            let max_intersaction_length = 16 + 16 * border_line.len();

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
                continue;
            } else {
                slope_x = slope_x.signum();
                slope_y = slope_y.signum();
            }

            // println!(
            //     "[I {current_label}] points: {}   slope: {slope_x}, {slope_y}",
            //     border_line.len()
            // );
            // println!("  {:?}", &border_line);

            // 2. Advance border points until crossing the intersected wire
            while let Some((x, y)) = border_line.pop() {
                let (mut pre_x, mut pre_y) = (x, y);

                for i in 1..=max_intersaction_length {
                    let rx = (x as i64 + (slope_x * i as f32) as i64) as u32;
                    let ry = (y as i64 + (slope_y * i as f32) as i64) as u32;

                    // If we are going in a diagonal, check connection before continuing
                    let mut ray_ended = false;

                    let mut ray_hit = |x: u32, y: u32| {
                        let color = *image.get_pixel(x, y);
                        let label = labels[get_index(x, y)];

                        if color != intersect_color {
                            ray_ended = true;
                        }

                        // If hit, connect wires "current_label" and "label"
                        if color == current_color && label != current_label {
                            // todo: search the real root !!!
                            let resolved_label = label_mapping[label as usize];
                            let resolved_current_label = label_mapping[current_label as usize];
                            if resolved_label != resolved_current_label {
                                let min = u32::min(resolved_label, resolved_current_label);
                                label_mapping[resolved_label as usize] = min;
                                label_mapping[resolved_current_label as usize] = min;
                                label_mapping[label as usize] = min;
                                label_mapping[current_label as usize] = min;
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

        // --- Merge labels of intersecting wires ---
        // Resove backwards dependencies
        for i in 0..label_mapping.len() {
            label_mapping[i] = label_mapping[label_mapping[i] as usize];
        }

        // Compact labels
        {
            // TODO use label_mapping[..i] as memory
            let mut compact_mapping: Vec<_> = (0..label_mapping.len() as u32).collect();
            current_label = 0;
            for i in 1..label_mapping.len() {
                if label_mapping[i] == i as u32 {
                    current_label += 1;
                    compact_mapping[i] = current_label;
                }
                // todo: change
                label_mapping[i] = compact_mapping[label_mapping[i] as usize];
            }
        }

        // Update labels
        for label in &mut labels {
            *label = label_mapping[*label as usize];
        }

        CircuitImage {
            image,
            labels,
            label_count: current_label,
        }
    }

    pub fn get_label(&self, x: u32, y: u32) -> u32 {
        self.labels[y as usize * self.image.width() as usize + x as usize]
    }
}
