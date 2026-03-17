use crate::{arg_parse_err::ArgParseErr, error::MagickError, image::Image};
use image::{DynamicImage, GrayImage, Luma};

#[derive(Debug, Clone, PartialEq)]
pub struct SauvolaConfig {
    pub window_size: u32,
    pub k: f32,
    pub r: f32,
}

impl Default for SauvolaConfig {
    fn default() -> Self {
        Self {
            window_size: 15,
            k: 0.2,
            r: 128.0,
        }
    }
}

impl SauvolaConfig {
    /// Parse from a comma-separated string of three values, or "default".
    /// Format: "window_size,k,r"
    /// Example: "15,0.2,128" or "default"
    pub fn parse_arg(s: &str) -> Result<Self, ArgParseErr> {
        let s = s.trim();
        if s.eq_ignore_ascii_case("default") {
            return Ok(Self::default());
        }

        let parts: Vec<&str> = s.split(',').collect();
        if parts.len() != 3 {
            return Err(ArgParseErr::with_msg(
                "sauvola requires 'default' or exactly 3 comma-separated values: \
                 window_size,k,r",
            ));
        }

        let window_size = parts[0].trim().parse::<u32>().map_err(|_| {
            ArgParseErr::with_msg("invalid window_size value (must be positive integer)")
        })?;

        let k = parts[1]
            .trim()
            .parse::<f32>()
            .map_err(|_| ArgParseErr::with_msg("invalid k value (must be float)"))?;

        let r = parts[2]
            .trim()
            .parse::<f32>()
            .map_err(|_| ArgParseErr::with_msg("invalid r value (must be float)"))?;

        Ok(Self { window_size, k, r })
    }
}

pub fn sauvola(image: &mut Image, config: &SauvolaConfig) -> Result<(), MagickError> {
    let input = image.pixels.to_luma8();
    let (width, height) = input.dimensions();
    let mut output = GrayImage::new(width, height);

    let w_int = (width + 1) as usize;
    let mut int_sum = vec![0u64; w_int * (height + 1) as usize];
    let mut int_sq_sum = vec![0u64; w_int * (height + 1) as usize];

    let idx = |x: u32, y: u32| -> usize { (y as usize * w_int) + x as usize };

    // 1. Build Integral Images
    for y in 0..height {
        for x in 0..width {
            let val = input.get_pixel(x, y)[0] as u64;
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

    // 2. Apply Sauvola Formula
    let half_w = config.window_size / 2;

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
            let variance = (sq_sum as f64 - (sum as f64 * sum as f64) / area) / area;
            let std_dev = if variance > 0.0 { variance.sqrt() } else { 0.0 };

            let threshold = mean * (1.0 + config.k as f64 * (std_dev / config.r as f64 - 1.0));

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
