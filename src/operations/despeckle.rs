use crate::{arg_parse_err::ArgParseErr, error::MagickError, image::Image};
use image::{DynamicImage, GrayImage, Luma};

#[derive(Debug, Clone, PartialEq)]
pub struct DespeckleConfig {
    pub kernel_size: u32,
}

impl Default for DespeckleConfig {
    fn default() -> Self {
        Self { kernel_size: 3 }
    }
}

impl DespeckleConfig {
    pub fn parse_arg(s: &str) -> Result<Self, ArgParseErr> {
        let s = s.trim();
        if s.eq_ignore_ascii_case("default") {
            return Ok(Self::default());
        }

        let kernel_size = s.parse::<u32>().map_err(|_| {
            ArgParseErr::with_msg("invalid despeckle size (must be a positive odd integer)")
        })?;

        if kernel_size % 2 == 0 {
            return Err(ArgParseErr::with_msg(
                "despeckle size should be an odd number (e.g., 3, 5)",
            ));
        }

        Ok(Self { kernel_size })
    }
}

pub fn despeckle(image: &mut Image, config: &DespeckleConfig) -> Result<(), MagickError> {
    // Fast path: A 1x1 median filter does nothing.
    if config.kernel_size <= 1 {
        return Ok(());
    }

    let input = image.pixels.to_luma8();
    let (width, height) = input.dimensions();
    let mut output = GrayImage::new(width, height);

    let half_k = config.kernel_size / 2;
    let capacity = (config.kernel_size * config.kernel_size) as usize;
    let mut window = Vec::with_capacity(capacity);

    for y in 0..height {
        for x in 0..width {
            window.clear();

            let x_start = x.saturating_sub(half_k);
            let x_end = (x + half_k).min(width - 1);
            let y_start = y.saturating_sub(half_k);
            let y_end = (y + half_k).min(height - 1);

            for ky in y_start..=y_end {
                for kx in x_start..=x_end {
                    window.push(input.get_pixel(kx, ky)[0]);
                }
            }

            // Sort to find the median
            window.sort_unstable();
            let median = window[window.len() / 2];

            output.put_pixel(x, y, Luma([median]));
        }
    }

    image.pixels = DynamicImage::ImageLuma8(output);
    Ok(())
}
