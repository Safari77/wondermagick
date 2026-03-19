use crate::{arg_parse_err::ArgParseErr, error::MagickError, image::Image, wm_err};
use bm3d_core::{bm3d_ring_artifact_removal, Bm3dConfig as CoreConfig, RingRemovalMode};
use image::{DynamicImage, GenericImageView, Luma, Rgba};
use ndarray::Array2;
use oklab::{oklab_to_srgb, srgb_to_oklab, Oklab, Rgb};

#[derive(Debug, Clone, PartialEq)]
pub struct Bm3dConfig {
    pub sigma_l: f32,
    pub sigma_a: f32,
    pub sigma_b: f32,
    pub patch_size: usize,
    pub step_size: usize,
    pub search_window: usize,
    pub max_matches: usize,
}

impl Default for Bm3dConfig {
    fn default() -> Self {
        Self {
            sigma_l: 0.05,
            sigma_a: 0.15,
            sigma_b: 0.15,
            patch_size: 8,
            step_size: 3,
            search_window: 24,
            max_matches: 16,
        }
    }
}

impl Bm3dConfig {
    /// Format: "sigma_l,sigma_a,sigma_b,patch_size,step_size,search_window,max_matches"
    /// Example: "0.05,0.15,0.15,8,3,24,16" or "default"
    pub fn parse_arg(s: &str) -> Result<Self, ArgParseErr> {
        let s = s.trim();
        if s.eq_ignore_ascii_case("default") {
            return Ok(Self::default());
        }

        let parts: Vec<&str> = s.split(',').collect();
        if parts.len() != 7 {
            return Err(ArgParseErr::with_msg(
                "bm3d requires 'default' or exactly 7 comma-separated values: \
                 sigma_l,sigma_a,sigma_b,patch_size,step_size,search_window,max_matches",
            ));
        }

        Ok(Self {
            sigma_l: parts[0]
                .trim()
                .parse()
                .map_err(|_| ArgParseErr::with_msg("invalid sigma_l"))?,
            sigma_a: parts[1]
                .trim()
                .parse()
                .map_err(|_| ArgParseErr::with_msg("invalid sigma_a"))?,
            sigma_b: parts[2]
                .trim()
                .parse()
                .map_err(|_| ArgParseErr::with_msg("invalid sigma_b"))?,
            patch_size: parts[3]
                .trim()
                .parse()
                .map_err(|_| ArgParseErr::with_msg("invalid patch_size"))?,
            step_size: parts[4]
                .trim()
                .parse()
                .map_err(|_| ArgParseErr::with_msg("invalid step_size"))?,
            search_window: parts[5]
                .trim()
                .parse()
                .map_err(|_| ArgParseErr::with_msg("invalid search_window"))?,
            max_matches: parts[6]
                .trim()
                .parse()
                .map_err(|_| ArgParseErr::with_msg("invalid max_matches"))?,
        })
    }
}

fn run_bm3d_channel(
    channel: &Array2<f32>,
    sigma: f32,
    config: &Bm3dConfig,
) -> Result<Array2<f32>, MagickError> {
    let core_config = CoreConfig {
        sigma_random: sigma,
        patch_size: config.patch_size,
        step_size: config.step_size,
        search_window: config.search_window,
        max_matches: config.max_matches,
        ..Default::default()
    };

    // Notice the updated 3-argument signature and .view() call
    bm3d_ring_artifact_removal(channel.view(), RingRemovalMode::Generic, &core_config)
        .map_err(|e| wm_err!("BM3D error: {}", e))
}

pub fn bm3d(image: &mut Image, config: &Bm3dConfig) -> Result<(), MagickError> {
    let (width, height) = image.pixels.dimensions();
    let w_usize = width as usize;
    let h_usize = height as usize;

    // Check if the image is grayscale or color
    if !image.pixels.color().has_color() {
        // --- GRAYSCALE PATH (1 Channel) ---
        let input = image.pixels.to_luma8();
        let mut luma_arr = Array2::<f32>::zeros((h_usize, w_usize));

        for (x, y, pixel) in input.enumerate_pixels() {
            luma_arr[[y as usize, x as usize]] = pixel[0] as f32 / 255.0;
        }

        let denoised_luma = run_bm3d_channel(&luma_arr, config.sigma_l, config)?;

        let mut output = image::GrayImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let val = (denoised_luma[[y as usize, x as usize]] * 255.0).clamp(0.0, 255.0) as u8;
                output.put_pixel(x, y, Luma([val]));
            }
        }
        image.pixels = DynamicImage::ImageLuma8(output);
    } else {
        // --- COLOR PATH (Oklab 3-Channel) ---
        let input = image.pixels.to_rgba8();

        let mut l_arr = Array2::<f32>::zeros((h_usize, w_usize));
        let mut a_arr = Array2::<f32>::zeros((h_usize, w_usize));
        let mut b_arr = Array2::<f32>::zeros((h_usize, w_usize));

        // 1. Convert to Oklab
        for (x, y, pixel) in input.enumerate_pixels() {
            let oklab = srgb_to_oklab(Rgb {
                r: pixel[0],
                g: pixel[1],
                b: pixel[2],
            });
            l_arr[[y as usize, x as usize]] = oklab.l;
            a_arr[[y as usize, x as usize]] = oklab.a;
            b_arr[[y as usize, x as usize]] = oklab.b;
        }

        // 2. Denoise independent channels
        let denoise_l = run_bm3d_channel(&l_arr, config.sigma_l, config)?;
        let denoise_a = run_bm3d_channel(&a_arr, config.sigma_a, config)?;
        let denoise_b = run_bm3d_channel(&b_arr, config.sigma_b, config)?;

        // 3. Reconstruct Rgba Image
        let mut output = image::RgbaImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let xu = x as usize;
                let yu = y as usize;

                let srgb = oklab_to_srgb(Oklab {
                    l: denoise_l[[yu, xu]],
                    a: denoise_a[[yu, xu]],
                    b: denoise_b[[yu, xu]],
                });

                // Preserve original Alpha channel
                let alpha = input.get_pixel(x, y)[3];
                output.put_pixel(x, y, Rgba([srgb.r, srgb.g, srgb.b, alpha]));
            }
        }
        image.pixels = DynamicImage::ImageRgba8(output);
    }

    Ok(())
}
