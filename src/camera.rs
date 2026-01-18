use eframe::egui::Vec2;

/// Camera used to project a 2d pixel art texture into
/// a 2d surface without texture x/y deformation.
#[derive(Debug, Copy, Clone)]
pub struct Camera {
    /// World position of the camera center (in texel space)
    pub position: Vec2,

    /// Uniform scale factor (surface pixels per texel)
    pub scale: f32,
}

impl Camera {
    /// Create a new camera
    pub fn new() -> Self {
        Self {
            position: Vec2::ZERO,
            scale: 1.0,
        }
    }

    /// Sets the base scale so that one texel maps to N surface pixels.
    /// This value is always uniform (no X/Y deformation).
    #[allow(dead_code)]
    pub fn set_surface_pixels_per_texel(&mut self, surface_pixels_per_texel: f32) {
        self.scale = surface_pixels_per_texel;
    }

    #[allow(dead_code)]
    pub fn surface_pixels_per_texel(&self) -> f32 {
        self.scale
    }

    pub fn texels_per_surface_pixel(&self) -> f32 {
        1. / self.scale
    }

    /// Zooms the camera uniformly around a center point in texel space.
    pub fn zoom_texture(&mut self, zoom_factor: f32, center: Vec2) {
        let zoom_factor = zoom_factor.max(0.0001);

        // Adjust position so the center point stays fixed
        self.position = center + (self.position - center) / zoom_factor;

        self.scale *= zoom_factor;
    }

    /// Zooms the camera uniformly around a center point in surface space.
    pub fn zoom_surface(&mut self, zoom_factor: f32, center: Vec2, surface_size: Vec2) {
        self.zoom_texture(zoom_factor, self.surface_to_texel(center, surface_size));
    }

    /// Texel space → Surface pixel space
    #[allow(dead_code)]
    pub fn texel_to_surface(&self, texel_pos: Vec2, surface_size: Vec2) -> Vec2 {
        (texel_pos - self.position) * self.scale + surface_size * 0.5
    }

    /// Surface pixel space → Texel space
    pub fn surface_to_texel(&self, surface_pos: Vec2, surface_size: Vec2) -> Vec2 {
        (surface_pos - surface_size * 0.5) / self.scale + self.position
    }

    /// Snaps scale to whole pixels for perfect pixel art rendering
    #[allow(dead_code)]
    pub fn snap_scale_to_pixels(&mut self) {
        self.scale = self.scale.round().max(1.0);
    }
}
