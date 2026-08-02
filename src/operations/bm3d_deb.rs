//! BM3D-DEB: deblurring / deconvolution.
//!
//! Unlike `-bm3d`, this operation only ever modifies **lightness**. The Oklab `a` and `b`
//! channels are carried through untouched, so hue and chroma are preserved by
//! construction and no parameter can produce a color cast. That is a deliberate
//! restriction: deconvolution is an ill-posed inverse, and applying it independently per
//! color channel lets small per-channel differences turn into color fringes, while a
//! chroma noise level that is large relative to the channel's range collapses that
//! channel entirely. Human vision is also far less sensitive to chroma sharpness than to
//! luminance sharpness, so there is little to gain from deconvolving chroma anyway.
//!
//! Consequently there is a single `sigma` (the luminance noise level) instead of the
//! per-channel `sigma_l,sigma_a,sigma_b` triple that `-bm3d` uses.

use crate::{arg_parse_err::ArgParseErr, error::MagickError, image::Image, wm_err};
use bm3d_core::{
    Bm3dDeblurConfig as CoreDeblurConfig, bm3d_deblur, boxcar_psf, estimate_white_noise_sigma,
    gaussian_psf,
};
use image::{DynamicImage, GenericImageView, Luma, Rgba};
use ndarray::Array2;
use oklab::{Oklab, Rgb, oklab_to_srgb, srgb_to_oklab};

/// Upper bound for a PSF standard deviation, in pixels.
/// Guards against absurd kernels (a Gaussian is truncated at 4 sigma).
const MAX_PSF_SIGMA: f32 = 50.0;

/// Upper bound for a box PSF edge length, in pixels.
const MAX_PSF_SIZE: usize = 255;

/// Upper bound for the de-ringing window radius, in pixels.
const MAX_DERING_RADIUS: usize = 8;

/// Floor for the automatically estimated noise level.
///
/// `sigma` does not only stand for sensor noise: it sets how much the inversion trusts
/// the observation, so it also has to absorb model error (8-bit quantization, JPEG
/// artifacts, a PSF that is only approximately Gaussian). One quantization step is the
/// smallest defensible value for 8-bit input.
const MIN_AUTO_SIGMA: f32 = 1.0 / 255.0;

/// Default stage-1 regularization, matching the `bm3d_core` reference value.
const DEFAULT_REG_RI: f32 = 1.0;

/// Default stage-2 regularization.
///
/// `bm3d_core` defaults to 5e-3, the value from the original BM3D-DEB paper, which
/// assumes an exactly known PSF and pure white noise. Photographs violate both, and the
/// mismatch shows up as ringing at high-contrast edges, so this operation defaults to
/// 1.0 - the statistically optimal Wiener regularization for the estimated noise level.
/// Raise it further for large PSFs (see the note in `parse_arg`).
const DEFAULT_REG_RWI: f32 = 1.0;

/// Point spread function describing the blur that BM3D-DEB should invert.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PsfSpec {
    /// Separable Gaussian blur, per-axis standard deviation in pixels.
    Gaussian { sigma_x: f32, sigma_y: f32 },
    /// Uniform (box) blur, size in pixels. Approximates defocus / detector binning.
    Box { width: usize, height: usize },
    /// No blur. BM3D-DEB then degenerates to plain colored-noise denoising.
    Delta,
}

impl PsfSpec {
    /// Format: "gauss:SIGMA", "gauss:SIGMA_XxSIGMA_Y", "box:N", "box:WxH", or "none".
    /// Examples: "gauss:1.2", "gauss:2.0x0.5", "box:3x3", "none"
    pub fn parse(s: &str) -> Result<Self, ArgParseErr> {
        let lowered = s.trim().to_lowercase();
        if lowered.is_empty() || lowered == "none" || lowered == "delta" {
            return Ok(Self::Delta);
        }

        let Some((kind, value)) = lowered.split_once(':') else {
            return Err(ArgParseErr::with_msg(
                "invalid psf: expected 'gauss:SIGMA', 'gauss:SIGMA_XxSIGMA_Y', \
                 'box:N', 'box:WxH' or 'none'",
            ));
        };

        match kind.trim() {
            "gauss" | "gaussian" | "g" => {
                let (sigma_x, sigma_y) = parse_pair_f32(value)?;
                if !sigma_x.is_finite() || !sigma_y.is_finite() {
                    return Err(ArgParseErr::with_msg("psf sigma must be finite"));
                }
                if sigma_x < 0.0 || sigma_y < 0.0 {
                    return Err(ArgParseErr::with_msg("psf sigma must be >= 0"));
                }
                if sigma_x > MAX_PSF_SIGMA || sigma_y > MAX_PSF_SIGMA {
                    return Err(ArgParseErr::with_msg("psf sigma is too large (max 50)"));
                }
                // A zero-sigma Gaussian on both axes is just a delta.
                if sigma_x == 0.0 && sigma_y == 0.0 {
                    return Ok(Self::Delta);
                }
                Ok(Self::Gaussian { sigma_x, sigma_y })
            }
            "box" | "boxcar" | "b" => {
                let (width, height) = parse_pair_usize(value)?;
                if width == 0 || height == 0 {
                    return Err(ArgParseErr::with_msg("psf box size must be >= 1"));
                }
                if width > MAX_PSF_SIZE || height > MAX_PSF_SIZE {
                    return Err(ArgParseErr::with_msg("psf box size is too large (max 255)"));
                }
                if width == 1 && height == 1 {
                    return Ok(Self::Delta);
                }
                Ok(Self::Box { width, height })
            }
            _ => Err(ArgParseErr::with_msg("unknown psf type: expected 'gauss', 'box' or 'none'")),
        }
    }

    /// Build the normalized PSF array expected by `bm3d_core`.
    ///
    /// Note the axis order: `bm3d_core` PSFs are indexed `[row, column]`, i.e.
    /// `[y, x]`, while the argument syntax uses the ImageMagick `WxH` convention.
    pub fn to_psf(self) -> Array2<f32> {
        match self {
            Self::Delta => boxcar_psf::<f32>(1, 1),
            Self::Gaussian { sigma_x, sigma_y } => gaussian_psf::<f32>(
                sigma_y.clamp(0.0, MAX_PSF_SIGMA),
                sigma_x.clamp(0.0, MAX_PSF_SIGMA),
            ),
            Self::Box { width, height } => {
                boxcar_psf::<f32>(height.clamp(1, MAX_PSF_SIZE), width.clamp(1, MAX_PSF_SIZE))
            }
        }
    }

    /// Window radius `(y, x)` used by the de-ringing clamp.
    ///
    /// One standard deviation is enough: overshoot sits right at the edge, so the window
    /// only has to be wide enough to reach the plateau on either side of it. A tighter
    /// window clamps harder.
    pub fn dering_radius(self) -> (usize, usize) {
        match self {
            Self::Delta => (0, 0),
            Self::Gaussian { sigma_x, sigma_y } => {
                (sigma_to_radius(sigma_y), sigma_to_radius(sigma_x))
            }
            Self::Box { width, height } => {
                ((height / 2).clamp(0, MAX_DERING_RADIUS), (width / 2).clamp(0, MAX_DERING_RADIUS))
            }
        }
    }
}

fn sigma_to_radius(sigma: f32) -> usize {
    if !(sigma > 0.0) {
        return 0;
    }
    (sigma.ceil() as usize).clamp(1, MAX_DERING_RADIUS)
}

fn parse_pair_f32(value: &str) -> Result<(f32, f32), ArgParseErr> {
    let value = value.trim();
    match value.split_once('x') {
        Some((first, second)) => Ok((parse_f32(first)?, parse_f32(second)?)),
        None => {
            let both = parse_f32(value)?;
            Ok((both, both))
        }
    }
}

fn parse_pair_usize(value: &str) -> Result<(usize, usize), ArgParseErr> {
    let value = value.trim();
    match value.split_once('x') {
        Some((first, second)) => Ok((parse_usize(first)?, parse_usize(second)?)),
        None => {
            let both = parse_usize(value)?;
            Ok((both, both))
        }
    }
}

fn parse_f32(value: &str) -> Result<f32, ArgParseErr> {
    value.trim().parse().map_err(|_| ArgParseErr::with_msg("invalid psf value: expected a number"))
}

fn parse_usize(value: &str) -> Result<usize, ArgParseErr> {
    value
        .trim()
        .parse()
        .map_err(|_| ArgParseErr::with_msg("invalid psf value: expected a whole number"))
}

/// Configuration for the BM3D-DEB (deblurring / deconvolution) operation.
#[derive(Debug, Clone, PartialEq)]
pub struct Bm3dDebConfig {
    /// Noise level of the *lightness* channel, on a 0-1 scale. 0 estimates it.
    /// This is a noise level, not a blur radius; the blur lives in `psf`.
    pub sigma: f32,
    pub patch_size: usize,
    pub step_size: usize,
    pub search_window: usize,
    pub max_matches: usize,
    pub psf: PsfSpec,
    /// Stage 1 (regularized inverse) regularization factor.
    pub reg_ri: f32,
    /// Stage 2 (regularized Wiener inverse) regularization factor.
    pub reg_rwi: f32,
    /// Strength of the de-ringing clamp, 0 (off) to 1 (full).
    pub deringing: f32,
}

impl Default for Bm3dDebConfig {
    fn default() -> Self {
        Self {
            sigma: 0.0,
            patch_size: 8,
            step_size: 3,
            search_window: 48,
            max_matches: 32,
            psf: PsfSpec::Gaussian { sigma_x: 1.0, sigma_y: 1.0 },
            reg_ri: DEFAULT_REG_RI,
            reg_rwi: DEFAULT_REG_RWI,
            deringing: 1.0,
        }
    }
}

impl Bm3dDebConfig {
    /// Format: "sigma,patch_size,step_size,search_window,max_matches,psf\[,reg_ri\[,reg_rwi\[,deringing\]\]\]"
    /// Example: "0,8,3,48,32,gauss:1.2" or "default"
    ///
    /// * `sigma` - lightness noise level (0-1 scale); 0 estimates it from the image.
    ///   It also absorbs model error, so raising it is the blunt safety knob: it
    ///   regularizes both inversion stages at once.
    /// * `psf` - the blur to invert; see [`PsfSpec::parse`].
    /// * `reg_ri` / `reg_rwi` - per-stage regularization. Ringing and blown highlights at
    ///   high-contrast edges mean the inversion is trusting the PSF too far: raise
    ///   `reg_rwi` (10-100 is reasonable for a PSF sigma much above ~1.2, where the
    ///   inverse filter's gain grows very quickly). Output too soft: lower it.
    /// * `deringing` - clamps the result to the local range of the input, 0 disables.
    ///
    /// Unlike `-bm3d` there is no per-channel sigma: only lightness is deblurred, so
    /// color cannot shift. There is also no Anscombe option, since the variance
    /// stabilizing transform does not commute with convolution and would invalidate the
    /// blur model.
    pub fn parse_arg(s: &str) -> Result<Self, ArgParseErr> {
        let s = s.trim();
        if s.eq_ignore_ascii_case("default") {
            return Ok(Self::default());
        }

        let parts: Vec<&str> = s.split(',').collect();
        // 6 required values, plus up to 3 optional tuning values.
        if parts.len() < 6 || parts.len() > 9 {
            return Err(ArgParseErr::with_msg(
                "bm3d-deb requires 'default' or 6-9 comma-separated values: \
                 sigma,patch_size,step_size,search_window,max_matches,psf\
                 [,reg_ri[,reg_rwi[,deringing]]]",
            ));
        }

        let defaults = Self::default();

        let sigma: f32 =
            parts[0].trim().parse().map_err(|_| ArgParseErr::with_msg("invalid sigma"))?;
        let patch_size: usize =
            parts[1].trim().parse().map_err(|_| ArgParseErr::with_msg("invalid patch_size"))?;
        let step_size: usize =
            parts[2].trim().parse().map_err(|_| ArgParseErr::with_msg("invalid step_size"))?;
        let search_window: usize =
            parts[3].trim().parse().map_err(|_| ArgParseErr::with_msg("invalid search_window"))?;
        let max_matches: usize =
            parts[4].trim().parse().map_err(|_| ArgParseErr::with_msg("invalid max_matches"))?;
        let psf = PsfSpec::parse(parts[5])?;

        let reg_ri = if parts.len() >= 7 {
            parts[6].trim().parse().map_err(|_| ArgParseErr::with_msg("invalid reg_ri"))?
        } else {
            defaults.reg_ri
        };
        let reg_rwi = if parts.len() >= 8 {
            parts[7].trim().parse().map_err(|_| ArgParseErr::with_msg("invalid reg_rwi"))?
        } else {
            defaults.reg_rwi
        };
        let deringing = if parts.len() >= 9 {
            parts[8].trim().parse().map_err(|_| ArgParseErr::with_msg("invalid deringing"))?
        } else {
            defaults.deringing
        };

        let config = Self {
            sigma,
            patch_size,
            step_size,
            search_window,
            max_matches,
            psf,
            reg_ri,
            reg_rwi,
            deringing,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), ArgParseErr> {
        if !self.sigma.is_finite() || self.sigma < 0.0 {
            return Err(ArgParseErr::with_msg("sigma must be finite and >= 0"));
        }
        if self.patch_size == 0 {
            return Err(ArgParseErr::with_msg("patch_size must be > 0"));
        }
        if self.step_size == 0 {
            return Err(ArgParseErr::with_msg("step_size must be > 0"));
        }
        if self.search_window == 0 {
            return Err(ArgParseErr::with_msg("search_window must be > 0"));
        }
        if self.max_matches == 0 {
            return Err(ArgParseErr::with_msg("max_matches must be > 0"));
        }
        if !self.reg_ri.is_finite() || self.reg_ri <= 0.0 {
            return Err(ArgParseErr::with_msg("reg_ri must be finite and > 0"));
        }
        if !self.reg_rwi.is_finite() || self.reg_rwi <= 0.0 {
            return Err(ArgParseErr::with_msg("reg_rwi must be finite and > 0"));
        }
        if !self.deringing.is_finite() || !(0.0..=1.0).contains(&self.deringing) {
            return Err(ArgParseErr::with_msg("deringing must be between 0 and 1"));
        }
        Ok(())
    }
}

/// Sliding-window minimum and maximum over a `(2*radius_y + 1) x (2*radius_x + 1)` box.
/// Separable: horizontal pass, then vertical pass over its output.
fn local_min_max(
    data: &Array2<f32>,
    radius_y: usize,
    radius_x: usize,
) -> (Array2<f32>, Array2<f32>) {
    let (rows, cols) = data.dim();
    let mut min_map = Array2::<f32>::zeros((rows, cols));
    let mut max_map = Array2::<f32>::zeros((rows, cols));
    if rows == 0 || cols == 0 {
        return (min_map, max_map);
    }

    // Horizontal pass
    for y in 0..rows {
        for x in 0..cols {
            let first = x.saturating_sub(radius_x);
            let last = (x + radius_x).min(cols - 1);
            let mut lo = f32::INFINITY;
            let mut hi = f32::NEG_INFINITY;
            for xx in first..=last {
                let value = data[[y, xx]];
                lo = lo.min(value);
                hi = hi.max(value);
            }
            min_map[[y, x]] = lo;
            max_map[[y, x]] = hi;
        }
    }

    // Vertical pass, reusing the horizontal results
    let min_h = min_map.clone();
    let max_h = max_map.clone();
    for y in 0..rows {
        let first = y.saturating_sub(radius_y);
        let last = (y + radius_y).min(rows - 1);
        for x in 0..cols {
            let mut lo = f32::INFINITY;
            let mut hi = f32::NEG_INFINITY;
            for yy in first..=last {
                lo = lo.min(min_h[[yy, x]]);
                hi = hi.max(max_h[[yy, x]]);
            }
            min_map[[y, x]] = lo;
            max_map[[y, x]] = hi;
        }
    }

    (min_map, max_map)
}

/// Constrain a deblurred channel to the local value range of its input.
///
/// Deconvolution overshoots at strong edges, and in 8-bit output that overshoot clips,
/// which is what shows up as bright halos next to high-contrast detail. Restricting each
/// output pixel to the range the input already takes nearby removes the halo without
/// touching the recovered edge slope. The cost is that features smaller than the PSF do
/// not get their peaks restored, which is the usual trade of any overshoot control.
fn apply_deringing(
    deblurred: &mut Array2<f32>,
    reference: &Array2<f32>,
    radius_y: usize,
    radius_x: usize,
    strength: f32,
) {
    if strength <= 0.0 || (radius_y == 0 && radius_x == 0) {
        return;
    }
    let (min_map, max_map) = local_min_max(reference, radius_y, radius_x);
    let (rows, cols) = deblurred.dim();
    for y in 0..rows {
        for x in 0..cols {
            let value = deblurred[[y, x]];
            let clamped = value.clamp(min_map[[y, x]], max_map[[y, x]]);
            deblurred[[y, x]] = value + strength * (clamped - value);
        }
    }
}

/// Deblur one lightness channel and apply the de-ringing clamp.
fn deblur_lightness(
    lightness: &Array2<f32>,
    config: &Bm3dDebConfig,
    psf: &Array2<f32>,
) -> Result<Array2<f32>, MagickError> {
    let sigma = if config.sigma > 0.0 {
        config.sigma
    } else {
        estimate_white_noise_sigma(lightness.view()).max(MIN_AUTO_SIGMA)
    };

    let core_config = CoreDeblurConfig::<f32> {
        sigma,
        reg_ri: config.reg_ri,
        reg_rwi: config.reg_rwi,
        patch_size: config.patch_size,
        step_size: config.step_size,
        search_window: config.search_window,
        max_matches: config.max_matches,
        ..Default::default()
    };

    let mut result = bm3d_deblur(lightness.view(), psf.view(), &core_config)
        .map_err(|e| wm_err!("BM3D-DEB error: {}", e))?;

    let (radius_y, radius_x) = config.psf.dering_radius();
    apply_deringing(&mut result, lightness, radius_y, radius_x, config.deringing);

    Ok(result)
}

pub fn bm3d_deb(image: &mut Image, config: &Bm3dDebConfig) -> Result<(), MagickError> {
    let (width, height) = image.pixels.dimensions();
    let w_usize = width as usize;
    let h_usize = height as usize;

    let psf = config.psf.to_psf();

    // Check if the image is grayscale or color
    if !image.pixels.color().has_color() {
        // --- GRAYSCALE PATH ---
        let input = image.pixels.to_luma8();
        let mut luma_arr = Array2::<f32>::zeros((h_usize, w_usize));

        for (x, y, pixel) in input.enumerate_pixels() {
            luma_arr[[y as usize, x as usize]] = pixel[0] as f32 / 255.0;
        }

        let deblurred = deblur_lightness(&luma_arr, config, &psf)?;

        let mut output = image::GrayImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let val = (deblurred[[y as usize, x as usize]] * 255.0).clamp(0.0, 255.0) as u8;
                output.put_pixel(x, y, Luma([val]));
            }
        }
        image.pixels = DynamicImage::ImageLuma8(output);
    } else {
        // --- COLOR PATH (Oklab, lightness only) ---
        let input = image.pixels.to_rgba8();

        let mut l_arr = Array2::<f32>::zeros((h_usize, w_usize));
        // a and b are stored only to be handed straight back to the inverse transform.
        let mut a_arr = Array2::<f32>::zeros((h_usize, w_usize));
        let mut b_arr = Array2::<f32>::zeros((h_usize, w_usize));

        // 1. Convert to Oklab
        for (x, y, pixel) in input.enumerate_pixels() {
            let oklab = srgb_to_oklab(Rgb { r: pixel[0], g: pixel[1], b: pixel[2] });
            l_arr[[y as usize, x as usize]] = oklab.l;
            a_arr[[y as usize, x as usize]] = oklab.a;
            b_arr[[y as usize, x as usize]] = oklab.b;
        }

        // 2. Deblur lightness only. Chroma is untouched, so hue and chroma survive
        //    exactly; the only color change possible is gamut clipping when a strongly
        //    brightened or darkened pixel is converted back to sRGB.
        let deblur_l = deblur_lightness(&l_arr, config, &psf)?;

        // 3. Reconstruct Rgba Image
        let mut output = image::RgbaImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let xu = x as usize;
                let yu = y as usize;

                let srgb = oklab_to_srgb(Oklab {
                    l: deblur_l[[yu, xu]],
                    a: a_arr[[yu, xu]],
                    b: b_arr[[yu, xu]],
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_default() {
        assert_eq!(Bm3dDebConfig::parse_arg("default").unwrap(), Bm3dDebConfig::default());
        assert_eq!(Bm3dDebConfig::parse_arg(" DEFAULT ").unwrap(), Bm3dDebConfig::default());
    }

    #[test]
    fn test_parse_required_fields() {
        let config = Bm3dDebConfig::parse_arg("0.02,8,3,24,16,gauss:1.5").unwrap();
        assert_eq!(config.sigma, 0.02);
        assert_eq!(config.patch_size, 8);
        assert_eq!(config.step_size, 3);
        assert_eq!(config.search_window, 24);
        assert_eq!(config.max_matches, 16);
        assert_eq!(config.psf, PsfSpec::Gaussian { sigma_x: 1.5, sigma_y: 1.5 });
        assert_eq!(config.reg_ri, DEFAULT_REG_RI);
        assert_eq!(config.reg_rwi, DEFAULT_REG_RWI);
        assert_eq!(config.deringing, 1.0);
    }

    #[test]
    fn test_parse_optional_fields() {
        let config = Bm3dDebConfig::parse_arg("0,8,4,24,16,box:3x3,2.0").unwrap();
        assert_eq!(config.reg_ri, 2.0);
        assert_eq!(config.reg_rwi, DEFAULT_REG_RWI);

        let config = Bm3dDebConfig::parse_arg("0,8,4,24,16,box:3x3,2.0,50").unwrap();
        assert_eq!(config.reg_rwi, 50.0);
        assert_eq!(config.deringing, 1.0);

        let config = Bm3dDebConfig::parse_arg("0,8,4,24,16,box:3x3,2.0,50,0").unwrap();
        assert_eq!(config.deringing, 0.0);
    }

    #[test]
    fn test_parse_rejects_bad_input() {
        // Too few / too many fields
        assert!(Bm3dDebConfig::parse_arg("0.01,8,4,24,16").is_err());
        assert!(Bm3dDebConfig::parse_arg("0.01,8,4,24,16,gauss:1,1,1,1,1").is_err());
        // Invalid or out-of-range values
        assert!(Bm3dDebConfig::parse_arg("abc,8,4,24,16,gauss:1").is_err());
        assert!(Bm3dDebConfig::parse_arg("-0.01,8,4,24,16,gauss:1").is_err());
        assert!(Bm3dDebConfig::parse_arg("0.01,8,0,24,16,gauss:1").is_err());
        assert!(Bm3dDebConfig::parse_arg("0.01,8,4,24,16,gauss:1,0").is_err());
        assert!(Bm3dDebConfig::parse_arg("0.01,8,4,24,16,gauss:1,1,1,2").is_err());
    }

    #[test]
    fn test_parse_psf_variants() {
        assert_eq!(PsfSpec::parse("none").unwrap(), PsfSpec::Delta);
        assert_eq!(PsfSpec::parse("gauss:0").unwrap(), PsfSpec::Delta);
        assert_eq!(PsfSpec::parse("box:1").unwrap(), PsfSpec::Delta);
        assert_eq!(
            PsfSpec::parse("Gauss:2.0x0.5").unwrap(),
            PsfSpec::Gaussian { sigma_x: 2.0, sigma_y: 0.5 }
        );
        assert_eq!(PsfSpec::parse(" box:5x3 ").unwrap(), PsfSpec::Box { width: 5, height: 3 });

        assert!(PsfSpec::parse("gauss").is_err());
        assert!(PsfSpec::parse("gauss:-1").is_err());
        assert!(PsfSpec::parse("gauss:999").is_err());
        assert!(PsfSpec::parse("box:0").is_err());
        assert!(PsfSpec::parse("disk:3").is_err());
    }

    #[test]
    fn test_psf_arrays_are_normalized_and_oriented() {
        let delta = PsfSpec::Delta.to_psf();
        assert_eq!(delta.dim(), (1, 1));
        assert!((delta[[0, 0]] - 1.0).abs() < 1e-6);

        // "WxH" in the argument maps to (rows, cols) == (H, W).
        let boxcar = PsfSpec::Box { width: 5, height: 3 }.to_psf();
        assert_eq!(boxcar.dim(), (3, 5));
        assert!((boxcar.iter().sum::<f32>() - 1.0).abs() < 1e-5);

        // Horizontal-only blur: sigma_y = 0 collapses the vertical axis.
        let horizontal = PsfSpec::Gaussian { sigma_x: 1.0, sigma_y: 0.0 }.to_psf();
        assert_eq!(horizontal.dim().0, 1);
        assert!(horizontal.dim().1 > 1);
        assert!((horizontal.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_dering_radius() {
        assert_eq!(PsfSpec::Delta.dering_radius(), (0, 0));
        assert_eq!(PsfSpec::Gaussian { sigma_x: 1.2, sigma_y: 0.0 }.dering_radius(), (0, 2));
        assert_eq!(PsfSpec::Box { width: 7, height: 3 }.dering_radius(), (1, 3));
        // Clamped to the maximum
        assert_eq!(
            PsfSpec::Gaussian { sigma_x: 40.0, sigma_y: 40.0 }.dering_radius(),
            (MAX_DERING_RADIUS, MAX_DERING_RADIUS)
        );
    }

    #[test]
    fn test_local_min_max() {
        let data = Array2::<f32>::from_shape_vec(
            (3, 3),
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        .unwrap();
        let (min_map, max_map) = local_min_max(&data, 1, 1);
        // Center sees the whole 3x3 block
        assert_eq!(min_map[[1, 1]], 0.0);
        assert_eq!(max_map[[1, 1]], 8.0);
        // Top-left corner window is clipped to the 2x2 block
        assert_eq!(min_map[[0, 0]], 0.0);
        assert_eq!(max_map[[0, 0]], 4.0);
    }

    #[test]
    fn test_deringing_clamps_overshoot() {
        // Step edge, with an overshoot spike next to it.
        let reference =
            Array2::<f32>::from_shape_fn((5, 5), |(_, x)| if x < 2 { 0.0 } else { 0.8 });
        let mut deblurred = reference.clone();
        deblurred[[2, 2]] = 1.6; // strong overshoot
        deblurred[[2, 1]] = -0.5; // undershoot on the dark side

        apply_deringing(&mut deblurred, &reference, 1, 1, 1.0);
        assert!((deblurred[[2, 2]] - 0.8).abs() < 1e-6);
        assert!((deblurred[[2, 1]] - 0.0).abs() < 1e-6);

        // Strength 0 leaves the input untouched.
        let mut untouched = reference.clone();
        untouched[[2, 2]] = 1.6;
        apply_deringing(&mut untouched, &reference, 1, 1, 0.0);
        assert_eq!(untouched[[2, 2]], 1.6);
    }

    #[test]
    fn test_deringing_preserves_valid_detail() {
        // A value already inside the local range must not be modified.
        let reference = Array2::<f32>::from_shape_fn((5, 5), |(y, x)| (y + x) as f32 / 8.0);
        let mut deblurred = reference.clone();
        apply_deringing(&mut deblurred, &reference, 1, 1, 1.0);
        for (a, b) in deblurred.iter().zip(reference.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
