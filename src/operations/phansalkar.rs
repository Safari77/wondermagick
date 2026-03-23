use crate::{arg_parse_err::ArgParseErr, error::MagickError, image::Image};
use image::{DynamicImage, GrayImage, Luma};

#[derive(Debug, Clone, PartialEq)]
pub struct PhansalkarConfig {
    pub window_size: u32,
    pub k: f32,
    pub r: f32,
    pub p: f32,
    pub q: f32,
}

impl Default for PhansalkarConfig {
    fn default() -> Self {
        Self {
            window_size: 15,
            k: 0.25,
            r: 0.5,
            p: 2.0,
            q: 10.0,
        }
    }
}

impl PhansalkarConfig {
    /// Parse from a comma-separated string of five values, or "default".
    /// Format: "window_size,k,r,p,q"
    /// Example: "15,0.25,0.5,2,10" or "default"
    pub fn parse_arg(s: &str) -> Result<Self, ArgParseErr> {
        let s = s.trim();
        if s.eq_ignore_ascii_case("default") {
            return Ok(Self::default());
        }

        let parts: Vec<&str> = s.split(',').collect();
        if parts.len() != 5 {
            return Err(ArgParseErr::with_msg(
                "phansalkar requires 'default' or exactly 5 comma-separated values: \
                 window_size,k,r,p,q",
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

        let p = parts[3]
            .trim()
            .parse::<f32>()
            .map_err(|_| ArgParseErr::with_msg("invalid p value (must be float)"))?;

        let q = parts[4]
            .trim()
            .parse::<f32>()
            .map_err(|_| ArgParseErr::with_msg("invalid q value (must be float)"))?;

        Ok(Self {
            window_size,
            k,
            r,
            p,
            q,
        })
    }
}

pub fn phansalkar(image: &mut Image, config: &PhansalkarConfig) -> Result<(), MagickError> {
    let input = image.pixels.to_luma8();
    let (width, height) = input.dimensions();
    let mut output = GrayImage::new(width, height);

    // Phansalkar operates on pixel values normalized to [0, 1].
    let norm = 1.0 / 255.0;

    let w_int = (width + 1) as usize;
    let mut int_sum = vec![0u64; w_int * (height + 1) as usize];
    let mut int_sq_sum = vec![0u64; w_int * (height + 1) as usize];

    let idx = |x: u32, y: u32| -> usize { (y as usize * w_int) + x as usize };

    // 1. Build Integral Images (in original 0-255 space; we normalize at query time)
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

    // 2. Apply Phansalkar Formula
    //    T(x,y) = mean * (1 + p * exp(-q * mean) + k * (std_dev / R - 1))
    //    where mean and std_dev are normalized to [0, 1].
    let half_w = config.window_size / 2;
    let k = config.k as f64;
    let r = config.r as f64;
    let p = config.p as f64;
    let q = config.q as f64;

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

            // Compute mean and std_dev in original 0-255 space, then normalize
            let mean_raw = sum as f64 / area;
            let variance_raw = (sq_sum as f64 - (mean_raw * mean_raw * area)) / area;
            let std_raw = if variance_raw > 0.0 {
                variance_raw.sqrt()
            } else {
                0.0
            };

            let mean = mean_raw * norm;
            let std_dev = std_raw * norm;

            let threshold = mean * (1.0 + p * (-q * mean).exp() + k * (std_dev / r - 1.0));

            // Compare in normalized space
            let pixel_val = input.get_pixel(x, y)[0] as f64 * norm;
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
