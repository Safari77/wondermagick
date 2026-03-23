use crate::{arg_parse_err::ArgParseErr, error::MagickError, image::Image};
use image::{DynamicImage, GrayImage, Luma};

#[derive(Debug, Clone, PartialEq)]
pub struct WolfJolionConfig {
    pub window_size: u32,
    pub k: f32,
}

impl Default for WolfJolionConfig {
    fn default() -> Self {
        Self {
            window_size: 15,
            k: 0.5,
        }
    }
}

impl WolfJolionConfig {
    /// Parse from a comma-separated string of two values, or "default".
    /// Format: "window_size,k"
    /// Example: "15,0.5" or "default"
    pub fn parse_arg(s: &str) -> Result<Self, ArgParseErr> {
        let s = s.trim();
        if s.eq_ignore_ascii_case("default") {
            return Ok(Self::default());
        }

        let parts: Vec<&str> = s.split(',').collect();
        if parts.len() != 2 {
            return Err(ArgParseErr::with_msg(
                "wolf-jolion requires 'default' or exactly 2 comma-separated values: \
                 window_size,k",
            ));
        }

        let window_size = parts[0].trim().parse::<u32>().map_err(|_| {
            ArgParseErr::with_msg("invalid window_size value (must be positive integer)")
        })?;

        let k = parts[1]
            .trim()
            .parse::<f32>()
            .map_err(|_| ArgParseErr::with_msg("invalid k value (must be float)"))?;

        Ok(Self { window_size, k })
    }
}

pub fn wolf_jolion(image: &mut Image, config: &WolfJolionConfig) -> Result<(), MagickError> {
    let input = image.pixels.to_luma8();
    let (width, height) = input.dimensions();
    let mut output = GrayImage::new(width, height);

    let w_int = (width + 1) as usize;
    let mut int_sum = vec![0u64; w_int * (height + 1) as usize];
    let mut int_sq_sum = vec![0u64; w_int * (height + 1) as usize];

    let idx = |x: u32, y: u32| -> usize { (y as usize * w_int) + x as usize };

    // 1. Build Integral Images and find the global minimum gray value (M)
    let mut global_min: u8 = 255;
    for y in 0..height {
        for x in 0..width {
            let pixel = input.get_pixel(x, y)[0];
            if pixel < global_min {
                global_min = pixel;
            }
            let val = pixel as u64;
            let sq_val = val * val;

            let top = int_sum[idx(x + 1, y)];
            let left = int_sum[idx(x, y + 1)];
            let top_left = int_sum[idx(x, y)];
            int_sum[idx(x + 1, y + 1)] = val + top + left - top_left;

            let sq_top = int_sq_sum[idx(x + 1, y)];
            let sq_left = int_sq_sum[idx(x, y + 1)];
            let sq_top_left = int_sq_sum[idx(x, y)];
            int_sq_sum[idx(x + 1, y + 1)] = sq_val + sq_top + sq_left - sq_top_left;
        }
    }

    let m = global_min as f64;

    // 2. First pass over all windows to find Rmax (maximum local standard deviation)
    let half_w = config.window_size / 2;
    let mut r_max: f64 = 0.0;

    for y in 0..height {
        for x in 0..width {
            let x1 = x.saturating_sub(half_w);
            let x2 = (x + half_w).min(width - 1);
            let y1 = y.saturating_sub(half_w);
            let y2 = (y + half_w).min(height - 1);

            let area = ((x2 - x1 + 1) * (y2 - y1 + 1)) as f64;

            let sum = int_sum[idx(x2 + 1, y2 + 1)] + int_sum[idx(x1, y1)]
                - int_sum[idx(x2 + 1, y1)]
                - int_sum[idx(x1, y2 + 1)];

            let sq_sum = int_sq_sum[idx(x2 + 1, y2 + 1)] + int_sq_sum[idx(x1, y1)]
                - int_sq_sum[idx(x2 + 1, y1)]
                - int_sq_sum[idx(x1, y2 + 1)];

            let mean = sum as f64 / area;
            let variance = (sq_sum as f64 - (mean * mean * area)) / area;
            let std_dev = if variance > 0.0 { variance.sqrt() } else { 0.0 };

            if std_dev > r_max {
                r_max = std_dev;
            }
        }
    }

    // Guard against degenerate images where Rmax is zero (uniform image)
    if r_max == 0.0 {
        r_max = 1.0;
    }

    // 3. Apply Wolf-Jolion formula:
    //    T(x,y) = (1 - k) * mean + k * M + k * (std_dev / Rmax) * (mean - M)
    //    which simplifies to:  mean - k * (mean - M) * (1 - std_dev / Rmax)
    //    but we use the expanded form for clarity.
    let k = config.k as f64;

    for y in 0..height {
        for x in 0..width {
            let x1 = x.saturating_sub(half_w);
            let x2 = (x + half_w).min(width - 1);
            let y1 = y.saturating_sub(half_w);
            let y2 = (y + half_w).min(height - 1);

            let area = ((x2 - x1 + 1) * (y2 - y1 + 1)) as f64;

            let sum = int_sum[idx(x2 + 1, y2 + 1)] + int_sum[idx(x1, y1)]
                - int_sum[idx(x2 + 1, y1)]
                - int_sum[idx(x1, y2 + 1)];

            let sq_sum = int_sq_sum[idx(x2 + 1, y2 + 1)] + int_sq_sum[idx(x1, y1)]
                - int_sq_sum[idx(x2 + 1, y1)]
                - int_sq_sum[idx(x1, y2 + 1)];

            let mean = sum as f64 / area;
            let variance = (sq_sum as f64 - (mean * mean * area)) / area;
            let std_dev = if variance > 0.0 { variance.sqrt() } else { 0.0 };

            let threshold = (1.0 - k) * mean + k * m + k * (std_dev / r_max) * (mean - m);

            let pixel_val = input.get_pixel(x, y)[0] as f64;
            if pixel_val > threshold {
                output.put_pixel(x, y, Luma([255]));
            } else {
                output.put_pixel(x, y, Luma([0]));
            }
        }
    }

    image.pixels = DynamicImage::ImageLuma8(output);
    Ok(())
}
