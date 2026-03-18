use crate::{arg_parse_err::ArgParseErr, error::MagickError, image::Image};
use image::{imageops, DynamicImage, GrayImage, Luma};

#[derive(Debug, Clone, PartialEq)]
pub struct NormalizeBackgroundConfig {
    pub radius: u32,
}

impl Default for NormalizeBackgroundConfig {
    fn default() -> Self {
        // 50 is usually a good radius to destroy text but keep the lighting gradients
        Self { radius: 50 }
    }
}

impl NormalizeBackgroundConfig {
    pub fn parse_arg(s: &str) -> Result<Self, ArgParseErr> {
        let s = s.trim();
        if s.eq_ignore_ascii_case("default") {
            return Ok(Self::default());
        }

        let radius = s.parse::<u32>().map_err(|_| {
            ArgParseErr::with_msg("invalid normalize radius (must be a positive integer)")
        })?;

        Ok(Self { radius })
    }
}

pub fn normalize_background(
    image: &mut Image,
    config: &NormalizeBackgroundConfig,
) -> Result<(), MagickError> {
    if config.radius == 0 {
        return Ok(());
    }

    let input = image.pixels.to_luma8();
    let (width, height) = input.dimensions();

    // STEP 1: Fast Background Estimation
    // Shrink the image by 4x. This makes the blur 16x faster.
    let small_w = (width / 4).max(1);
    let small_h = (height / 4).max(1);

    let small_img = imageops::resize(&input, small_w, small_h, imageops::FilterType::Nearest);

    // Blur the tiny image. We divide the radius by 4 to match the new scale.
    let blur_sigma = (config.radius as f32) / 4.0;
    let blurred_small = imageops::blur(&small_img, blur_sigma);

    // Blow it back up to full size to act as our Illumination Map
    let bg_map = imageops::resize(
        &blurred_small,
        width,
        height,
        imageops::FilterType::Triangle,
    );

    // STEP 2: Image Arithmetic (Division)
    let mut output = GrayImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let orig = input.get_pixel(x, y)[0] as f32;
            let bg = bg_map.get_pixel(x, y)[0] as f32;

            // Reflectance = (Original / Background) * 255
            let norm = if bg > 0.0 { (orig / bg) * 255.0 } else { orig };

            output.put_pixel(x, y, Luma([norm.clamp(0.0, 255.0) as u8]));
        }
    }

    image.pixels = DynamicImage::ImageLuma8(output);
    Ok(())
}
