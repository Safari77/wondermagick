use crate::{arg_parse_err::ArgParseErr, error::MagickError, image::Image, wm_err};
use gainforge::{
    create_tone_mapper_rgb, create_tone_mapper_rgb16, create_tone_mapper_rgba,
    create_tone_mapper_rgba16, AgxCustomLook, AgxLook, CommonToneMapperParameters,
    FilmicSplineParameters, ForgeError, GainHdrMetadata, GamutClipping, JzazbzToneMapperParameters,
    MappingColorSpace, RgbToneMapperParameters, ToneMappingMethod,
};
use image::{DynamicImage, ImageBuffer, Rgb as ImgRgb, RgbImage, Rgba as ImgRgba, RgbaImage};
use moxcms::{ColorProfile, Rgb};
use std::borrow::Cow;

// -----------------------------------------------------------------------------
// Configuration data-model
// -----------------------------------------------------------------------------

/// All knobs for the Blender Filmic / Darktable-style spline tone mapper.
/// Defaults follow the `Default` impl in `gainforge::FilmicSplineParameters`.
#[derive(Debug, Clone, PartialEq)]
pub struct FilmicSplineConfig {
    pub output_power: f32,
    pub latitude: f32,           // 0.01 .. 99     -- 33.0
    pub white_point_source: f32, // 0.1  .. 16     -- 3.0
    pub black_point_source: f32, // -16  .. -0.1   -- -8.0
    pub contrast: f32,           // 0    .. 5      -- 1.18
    pub black_point_target: f32, // 0    .. 20     -- 0.01517634
    pub grey_point_target: f32,  // 1    .. 50     -- 18.45
    pub white_point_target: f32, // 0    .. 1600   -- 100.0
    pub balance: f32,            // -50  .. 50     -- 0.0
    pub saturation: f32,         // -200 .. 200    -- 0.0
}

impl Default for FilmicSplineConfig {
    fn default() -> Self {
        // Mirror gainforge's own defaults so we never silently drift from the
        // crate when it retunes them.
        let d = FilmicSplineParameters::default();
        Self {
            output_power: d.output_power,
            latitude: d.latitude,
            white_point_source: d.white_point_source,
            black_point_source: d.black_point_source,
            contrast: d.contrast,
            black_point_target: d.black_point_target,
            grey_point_target: d.grey_point_target,
            white_point_target: d.white_point_target,
            balance: d.balance,
            saturation: d.saturation,
        }
    }
}

impl FilmicSplineConfig {
    /// Range-checks every knob against the documented gainforge limits so an
    /// out-of-range value fails at parse time instead of producing garbage or
    /// panicking deep inside the mapper.
    fn validate(&self) -> Result<(), ArgParseErr> {
        check_positive_finite(
            self.output_power,
            "fs_output_power must be positive and finite",
        )?;
        check_range(
            self.latitude,
            0.01,
            99.0,
            "fs_latitude out of range (0.01 .. 99)",
        )?;
        check_range(
            self.white_point_source,
            0.1,
            16.0,
            "fs_white_source out of range (0.1 .. 16)",
        )?;
        check_range(
            self.black_point_source,
            -16.0,
            -0.1,
            "fs_black_source out of range (-16 .. -0.1)",
        )?;
        check_range(self.contrast, 0.0, 5.0, "fs_contrast out of range (0 .. 5)")?;
        check_range(
            self.black_point_target,
            0.0,
            20.0,
            "fs_black_target out of range (0 .. 20)",
        )?;
        check_range(
            self.grey_point_target,
            1.0,
            50.0,
            "fs_grey_target out of range (1 .. 50)",
        )?;
        check_range(
            self.white_point_target,
            0.0,
            1600.0,
            "fs_white_target out of range (0 .. 1600)",
        )?;
        check_range(
            self.balance,
            -50.0,
            50.0,
            "fs_balance out of range (-50 .. 50)",
        )?;
        check_range(
            self.saturation,
            -200.0,
            200.0,
            "fs_saturation out of range (-200 .. 200)",
        )?;
        Ok(())
    }
}

/// Pre-parsed form of `AgxCustomLook`: each per-channel value may be either a
/// single f32 (broadcast) or `r:g:b` triple.  Conversion to the gainforge
/// `Rgb<f32>` type happens at call time.
#[derive(Debug, Clone, PartialEq)]
pub struct AgxCustomLookConfig {
    pub slope: [f32; 3],
    pub power: [f32; 3],
    pub saturation: [f32; 3],
    pub offset: [f32; 3],
}

impl Default for AgxCustomLookConfig {
    fn default() -> Self {
        Self {
            slope: [1.0, 1.0, 1.0],
            power: [1.0, 1.0, 1.0],
            saturation: [1.0, 1.0, 1.0],
            offset: [0.0, 0.0, 0.0],
        }
    }
}

impl AgxCustomLookConfig {
    /// gainforge doesn't publish hard ranges for the ASC-CDL-style AgX knobs,
    /// so we apply conservative sanity limits: every value must be finite,
    /// slope/saturation non-negative, and power strictly positive (a zero or
    /// negative exponent is degenerate). Offset may be any finite value.
    fn validate(&self) -> Result<(), ArgParseErr> {
        for &v in self.slope.iter() {
            if !v.is_finite() || v < 0.0 {
                return Err(ArgParseErr::with_msg(
                    "agx_slope values must be finite and >= 0",
                ));
            }
        }
        for &v in self.power.iter() {
            if !v.is_finite() || v <= 0.0 {
                return Err(ArgParseErr::with_msg(
                    "agx_power values must be finite and > 0",
                ));
            }
        }
        for &v in self.saturation.iter() {
            if !v.is_finite() || v < 0.0 {
                return Err(ArgParseErr::with_msg(
                    "agx_saturation values must be finite and >= 0",
                ));
            }
        }
        for &v in self.offset.iter() {
            if !v.is_finite() {
                return Err(ArgParseErr::with_msg("agx_offset values must be finite"));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TonemapConfig {
    /// `None` means "auto".  Until PNG-chunk plumbing lands we fall back to
    /// BT.2020 PQ full range.
    pub cicp: Option<[u8; 4]>,
    /// Source content peak luminance in nits (cd/m^2).
    pub nits: f32,
    /// Output (display) target peak luminance in nits, defaults to 100 (SDR).
    pub display_max_brightness: f32,
    /// Tone-mapping algorithm (case-insensitive).
    pub method: String,
    /// Linear exposure multiplier applied *during* tone mapping.
    pub exposure: f32,
    /// Which color space the mapper operates in: `rgb`, `yrg`, or `jzazbz`.
    pub color_space: String,
    /// `noclip` (default) or `clip`.
    pub gamut_clipping: String,
    /// `ExtendedReinhard` max luma.
    pub max_luma: f32,
    /// `Jzazbz` content brightness in nits; if `None` we reuse `nits`.
    pub content_brightness: Option<f32>,
    /// Filmic-spline parameters.
    pub filmic_spline: FilmicSplineConfig,
    /// `AgxLook` preset name (`default`/`punchy`/`golden`/`custom`).
    pub agx_look: String,
    /// Per-channel AgX custom look values.
    pub agx_custom: AgxCustomLookConfig,
}

impl Default for TonemapConfig {
    fn default() -> Self {
        Self {
            cicp: Some([9, 16, 0, 1]),     // BT.2020 PQ Full Range
            nits: 1000.0,                  // default content peak nits
            display_max_brightness: 100.0, // SDR target
            method: "itu2408".to_string(),
            exposure: 1.0,
            color_space: "yrg".to_string(),
            gamut_clipping: "noclip".to_string(),
            max_luma: 3.0,
            content_brightness: None,
            filmic_spline: FilmicSplineConfig::default(),
            agx_look: "default".to_string(),
            agx_custom: AgxCustomLookConfig::default(),
        }
    }
}

impl TonemapConfig {
    /// Parse from string format:
    ///
    ///     `cicp=9,16,0,1,nits=1000,tonemapping=itu2408,exposure=1.2,colorspace=yrg`
    ///
    /// Or the literal `default`.
    pub fn parse_arg(s: &str) -> Result<Self, ArgParseErr> {
        let s = s.trim();
        if s.eq_ignore_ascii_case("default") {
            return Ok(Self::default());
        }

        let mut config = Self::default();
        let mut parts = s.split(',');

        while let Some(part) = parts.next() {
            let part = part.trim();

            if let Some(val) = part.strip_prefix("cicp=") {
                if val.eq_ignore_ascii_case("auto") {
                    config.cicp = None;
                } else {
                    let parse_byte = |v: &str| -> Result<u8, ArgParseErr> {
                        v.trim().parse::<u8>().map_err(|_| {
                            ArgParseErr::with_msg("invalid cicp value (must be integer 0-255)")
                        })
                    };
                    let p1 = parse_byte(val)?;
                    let p2_val = parts
                        .next()
                        .ok_or_else(|| ArgParseErr::with_msg("cicp requires 4 values"))?;
                    let p2 = parse_byte(p2_val)?;
                    let p3_val = parts
                        .next()
                        .ok_or_else(|| ArgParseErr::with_msg("cicp requires 4 values"))?;
                    let p3 = parse_byte(p3_val)?;
                    let p4_val = parts
                        .next()
                        .ok_or_else(|| ArgParseErr::with_msg("cicp requires 4 values"))?;
                    let p4 = parse_byte(p4_val)?;
                    let cicp = [p1, p2, p3, p4];
                    // Reject unsupported/ill-formed cICP now, before we spend
                    // CPU decoding an image we can't tone-map anyway.
                    validate_cicp_arg(cicp)?;
                    config.cicp = Some(cicp);
                }
            } else if let Some(val) = part.strip_prefix("nits=") {
                config.nits = parse_f32(val, "invalid nits value (must be float)")?;
            } else if let Some(val) = part
                .strip_prefix("tonemapping=")
                .or_else(|| part.strip_prefix("method="))
            {
                config.method = val.to_lowercase();
            } else if let Some(val) = part.strip_prefix("exposure=") {
                config.exposure = parse_f32(val, "invalid exposure value (must be float)")?;
            } else if let Some(val) = part
                .strip_prefix("colorspace=")
                .or_else(|| part.strip_prefix("cs="))
            {
                config.color_space = val.to_lowercase();
            } else if let Some(val) = part
                .strip_prefix("gamut_clipping=")
                .or_else(|| part.strip_prefix("gc="))
            {
                config.gamut_clipping = val.to_lowercase();
            } else if let Some(val) = part.strip_prefix("max_luma=") {
                config.max_luma = parse_f32(val, "invalid max_luma value (must be float)")?;
            } else if let Some(val) = part
                .strip_prefix("content_brightness=")
                .or_else(|| part.strip_prefix("cb="))
            {
                config.content_brightness = Some(parse_f32(
                    val,
                    "invalid content_brightness value (must be float)",
                )?);
            } else if let Some(val) = part
                .strip_prefix("display_max_brightness=")
                .or_else(|| part.strip_prefix("display_nits="))
            {
                config.display_max_brightness =
                    parse_f32(val, "invalid display_max_brightness value (must be float)")?;
            // --- Filmic Spline parameters ---
            } else if let Some(val) = part.strip_prefix("fs_output_power=") {
                config.filmic_spline.output_power =
                    parse_f32(val, "invalid fs_output_power value (must be float)")?;
            } else if let Some(val) = part.strip_prefix("fs_latitude=") {
                config.filmic_spline.latitude =
                    parse_f32(val, "invalid fs_latitude value (must be float)")?;
            } else if let Some(val) = part.strip_prefix("fs_white_source=") {
                config.filmic_spline.white_point_source =
                    parse_f32(val, "invalid fs_white_source value (must be float)")?;
            } else if let Some(val) = part.strip_prefix("fs_black_source=") {
                config.filmic_spline.black_point_source =
                    parse_f32(val, "invalid fs_black_source value (must be float)")?;
            } else if let Some(val) = part.strip_prefix("fs_contrast=") {
                config.filmic_spline.contrast =
                    parse_f32(val, "invalid fs_contrast value (must be float)")?;
            } else if let Some(val) = part.strip_prefix("fs_black_target=") {
                config.filmic_spline.black_point_target =
                    parse_f32(val, "invalid fs_black_target value (must be float)")?;
            } else if let Some(val) = part.strip_prefix("fs_grey_target=") {
                config.filmic_spline.grey_point_target =
                    parse_f32(val, "invalid fs_grey_target value (must be float)")?;
            } else if let Some(val) = part.strip_prefix("fs_white_target=") {
                config.filmic_spline.white_point_target =
                    parse_f32(val, "invalid fs_white_target value (must be float)")?;
            } else if let Some(val) = part.strip_prefix("fs_balance=") {
                config.filmic_spline.balance =
                    parse_f32(val, "invalid fs_balance value (must be float)")?;
            } else if let Some(val) = part.strip_prefix("fs_saturation=") {
                config.filmic_spline.saturation =
                    parse_f32(val, "invalid fs_saturation value (must be float)")?;
            // --- Agx look & custom overrides ---
            } else if let Some(val) = part.strip_prefix("agx_look=") {
                config.agx_look = val.to_lowercase();
            } else if let Some(val) = part.strip_prefix("agx_slope=") {
                config.agx_custom.slope = parse_rgb3(val, "invalid agx_slope value")?;
            } else if let Some(val) = part.strip_prefix("agx_power=") {
                config.agx_custom.power = parse_rgb3(val, "invalid agx_power value")?;
            } else if let Some(val) = part.strip_prefix("agx_saturation=") {
                config.agx_custom.saturation = parse_rgb3(val, "invalid agx_saturation value")?;
            } else if let Some(val) = part.strip_prefix("agx_offset=") {
                config.agx_custom.offset = parse_rgb3(val, "invalid agx_offset value")?;
            } else {
                return Err(ArgParseErr::with_msg(
                    "unknown tonemap parameter. Expected cicp, nits, tonemapping, exposure, colorspace, \
                     gamut_clipping, max_luma, content_brightness, display_max_brightness, \
                     fs_* (filmic spline), agx_look, agx_slope, agx_power, agx_saturation, or agx_offset",
                ));
            }
        }

        // Sanity-check the most important numericals.
        if !config.nits.is_finite() || config.nits <= 0.0 {
            return Err(ArgParseErr::with_msg(
                "nits must be a positive, finite number",
            ));
        }
        if !config.exposure.is_finite() || config.exposure <= 0.0 {
            return Err(ArgParseErr::with_msg(
                "exposure must be a positive, finite number",
            ));
        }
        if !config.display_max_brightness.is_finite() || config.display_max_brightness <= 0.0 {
            return Err(ArgParseErr::with_msg(
                "display_max_brightness must be a positive, finite number",
            ));
        }
        if config.max_luma <= 0.0 || !config.max_luma.is_finite() {
            return Err(ArgParseErr::with_msg(
                "max_luma must be positive and finite",
            ));
        }
        if let Some(cb) = config.content_brightness
            && (!cb.is_finite() || cb <= 0.0) {
                return Err(ArgParseErr::with_msg(
                    "content_brightness must be positive and finite",
                ));
            }
        // Filmic-spline and AgX knobs feed straight into gainforge, so range-check
        // them here too rather than trusting the mapper to handle garbage.
        config.filmic_spline.validate()?;
        config.agx_custom.validate()?;
        Ok(config)
    }
}

// -----------------------------------------------------------------------------
// Small parsing helpers
// -----------------------------------------------------------------------------

fn parse_f32(val: &str, err_label: &'static str) -> Result<f32, ArgParseErr> {
    val.trim()
        .parse::<f32>()
        .map_err(|_| ArgParseErr::with_msg(err_label))
}

/// Parse a 3-tuple either as a broadcast `v` or as `r:g:b`.
fn parse_rgb3(val: &str, err_label: &'static str) -> Result<[f32; 3], ArgParseErr> {
    let pieces: Vec<&str> = val.split(':').collect();
    match pieces.len() {
        1 => {
            let v = parse_f32(pieces[0], err_label)?;
            Ok([v, v, v])
        }
        3 => {
            let r = parse_f32(pieces[0], err_label)?;
            let g = parse_f32(pieces[1], err_label)?;
            let b = parse_f32(pieces[2], err_label)?;
            Ok([r, g, b])
        }
        _ => Err(ArgParseErr::with_msg(
            "expected 1 broadcast value or 3 colon-separated values (r:g:b)",
        )),
    }
}

/// Inclusive `[lo, hi]` range check that also rejects NaN/inf.
fn check_range(v: f32, lo: f32, hi: f32, err: &'static str) -> Result<(), ArgParseErr> {
    if !v.is_finite() || v < lo || v > hi {
        return Err(ArgParseErr::with_msg(err));
    }
    Ok(())
}

/// Rejects NaN/inf and anything `<= 0`.
fn check_positive_finite(v: f32, err: &'static str) -> Result<(), ArgParseErr> {
    if !v.is_finite() || v <= 0.0 {
        return Err(ArgParseErr::with_msg(err));
    }
    Ok(())
}

/// Parse-time cICP validation. Keeps the accepted (primaries, transfer) set in
/// sync with `get_color_profile` below so a bad profile errors before we bother
/// decoding an image. `auto` (i.e. `None`) is validated later once resolved.
fn validate_cicp_arg(cicp: [u8; 4]) -> Result<(), ArgParseErr> {
    if cicp[2] != 0 {
        return Err(ArgParseErr::with_msg(
            "cICP matrix coefficients must be 0 (RGB); YCbCr/ICtCp is not supported",
        ));
    }
    if cicp[3] > 1 {
        return Err(ArgParseErr::with_msg(
            "cICP video full range flag must be 0 (narrow) or 1 (full)",
        ));
    }
    // Only the primaries/transfer pairs handled by get_color_profile are valid.
    match (cicp[0], cicp[1]) {
        (9, 16) | (9, 18) | (1, 13) | (12, 16) => Ok(()),
        _ => Err(ArgParseErr::with_msg(
            "unsupported cICP primaries/transfer. Supported: 9-16 (BT.2020 PQ), \
             9-18 (BT.2020 HLG), 12-16 (Display P3 PQ), 1-13 (sRGB)",
        )),
    }
}

// -----------------------------------------------------------------------------
// Color space + tone-mapping enum mirroring (validated once up front)
// -----------------------------------------------------------------------------

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ColorSpaceKind {
    Rgb,
    Yrg,
    Jzazbz,
}

/// Grayscale input variants we tone-map through the RGB path and then collapse
/// back to luma, so grayscale in yields grayscale out.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum GrayKind {
    L8,
    La8,
    L16,
    La16,
}

// -----------------------------------------------------------------------------
// cICP helpers (unchanged from the previous version)
// -----------------------------------------------------------------------------

/// Validates the cICP bytes we do not otherwise interpret.
fn validate_cicp(cicp: [u8; 4]) -> Result<(), MagickError> {
    // PNG (and our RGB pipeline) only supports Matrix Coefficients = 0 (identity / RGB).
    // Any other value would mean YCbCr/ICtCp-coded samples, which we cannot have here.
    if cicp[2] != 0 {
        return Err(wm_err!(
            "unsupported cICP matrix coefficients {} (must be 0 / RGB)",
            cicp[2]
        ));
    }
    // Video Full Range Flag is a boolean: 0 = narrow (16-235 scaled), 1 = full.
    if cicp[3] > 1 {
        return Err(wm_err!(
            "invalid cICP video full range flag {} (must be 0 or 1)",
            cicp[3]
        ));
    }
    Ok(())
}

/// Matches cICP integer profiles to moxcms ColorProfile constructors.
fn get_color_profile(cicp: [u8; 4]) -> Result<ColorProfile, MagickError> {
    // cicp = [Color Primaries, Transfer Characteristics, Matrix Coefficients, Range Flag]
    match (cicp[0], cicp[1]) {
        (9, 16) => Ok(ColorProfile::new_bt2020_pq()),
        (9, 18) => Ok(ColorProfile::new_bt2020_hlg()),
        (1, 13) => Ok(ColorProfile::new_srgb()),
        (12, 16) => Ok(ColorProfile::new_display_p3_pq()),
       // TODO: moxcms 0.8 ships no ready-made Display P3 HLG profile. Build one via
       // ColorProfile::new_from_cicp once we map raw cICP bytes to the moxcms CICP enums.
        (12, 18) => Err(wm_err!(
            "Display P3 + HLG (cICP 12-18) is not supported yet"
        )),
        _ => Err(wm_err!(
            "Unsupported cICP profile: primaries={}, transfer={}. Common: 9-16 (BT.2020 PQ), 9-18 (BT.2020 HLG), 12-16 (Display P3 PQ), 1-13 (sRGB)",
            cicp[0],
            cicp[1]
        )),
    }
}

/// Expands H.273 narrow (broadcast) range color samples to full range in place.
/// Narrow range stores values in [16, 235] * 2^(n-8) for an n-bit image.
/// `channels` is the interleave stride; `color_channels` is how many leading
/// channels are colour (1 for luma, 3 for RGB). Alpha is always full range in
/// PNG, so trailing channels beyond `color_channels` are left untouched.
fn expand_narrow_lane_u8(data: &mut [u8], channels: usize, color_channels: usize) {
    for px in data.chunks_exact_mut(channels) {
        for v in px[..color_channels].iter_mut() {
            let x = (i32::from(*v) - 16).clamp(0, 219) as u32;
            *v = ((x * 255 + 109) / 219) as u8; // +109 rounds to nearest
        }
    }
}

fn expand_narrow_lane_u16(data: &mut [u16], channels: usize, color_channels: usize) {
    const LO: i32 = 16 << 8; // 4096
    const SPAN: u64 = 219 << 8; // 56064
    for px in data.chunks_exact_mut(channels) {
        for v in px[..color_channels].iter_mut() {
            let x = (i32::from(*v) - LO).clamp(0, SPAN as i32) as u64;
            *v = ((x * 65535 + SPAN / 2) / SPAN) as u16;
        }
    }
}

fn expand_narrow_range(image: &mut Image) -> Result<(), MagickError> {
    match &mut image.pixels {
        DynamicImage::ImageRgb8(buf) => expand_narrow_lane_u8(buf, 3, 3),
        DynamicImage::ImageRgba8(buf) => expand_narrow_lane_u8(buf, 4, 3),
        DynamicImage::ImageLuma8(buf) => expand_narrow_lane_u8(buf, 1, 1),
        DynamicImage::ImageLumaA8(buf) => expand_narrow_lane_u8(buf, 2, 1),
        DynamicImage::ImageRgb16(buf) => expand_narrow_lane_u16(buf, 3, 3),
        DynamicImage::ImageRgba16(buf) => expand_narrow_lane_u16(buf, 4, 3),
        DynamicImage::ImageLuma16(buf) => expand_narrow_lane_u16(buf, 1, 1),
        DynamicImage::ImageLumaA16(buf) => expand_narrow_lane_u16(buf, 2, 1),
        _ => {
            return Err(wm_err!(
                "narrow range expansion is not supported for this pixel format"
            ))
        }
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// Tone-mapping core
// -----------------------------------------------------------------------------

/// Runs a gainforge tone mapper row by row over an interleaved pixel buffer.
fn map_rows<T: Copy + Default>(
    src: &[T],
    row_len: usize,
    mut lane: impl FnMut(&[T], &mut [T]) -> Result<(), ForgeError>,
) -> Result<Vec<T>, MagickError> {
    let mut dst = vec![T::default(); src.len()];
    for (y, (src_row, dst_row)) in src
        .chunks_exact(row_len)
        .zip(dst.chunks_exact_mut(row_len))
        .enumerate()
    {
        lane(src_row, dst_row).map_err(|e| wm_err!("Tone mapping failed on row {}: {}", y, e))?;
    }
    Ok(dst)
}

pub fn tonemap(image: &mut Image, config: &TonemapConfig) -> Result<(), MagickError> {
    // 1. Determine Source Color Profile
    // "auto" fallback: Default to BT2020 PQ until PNG chunk metadata extraction is implemented
    let cicp = config.cicp.unwrap_or([9, 16, 0, 1]);
    validate_cicp(cicp)?;
    let src_profile = get_color_profile(cicp)?;
    // Narrow (broadcast) range data must be expanded to full range before the
    // transfer function is undone, otherwise every level is decoded wrong.
    if cicp[3] == 0 {
        expand_narrow_range(image)?;
    }

    // sRGB-tagged input is already SDR; nothing useful to do. (Grayscale stays
    // grayscale here because we never touch image.pixels on this path.)
    if cicp[1] == 13 {
        return Ok(());
    }

    let dst_profile = ColorProfile::new_srgb();

    // --- 2. Select Tone Mapping Method --------------------------------------
    let gain_metadata = GainHdrMetadata {
        content_max_brightness: config.nits,
        display_max_brightness: config.display_max_brightness,
    };

    let tone_mapping_method = match config.method.as_str() {
        // ITU-R BT.2408 broadcast tone curve (default).
        "itu2408" | "rec2408" | "bt2408" | "default" => ToneMappingMethod::Itu2408(gain_metadata),

        // Tuned Reinhard (fast/accurate, tuned for typical HDR sources).
        "tuned_reinhard" | "tuned" => ToneMappingMethod::TunedReinhard(gain_metadata),

        // Plain Erik Reinhard operator.
        "reinhard" => ToneMappingMethod::Reinhard,

        // Same as Reinhard but scaled to the full dynamic range of the image
        // (max_luma is in linear scene-referred units, e.g. 3.0 = ~3x nominal exposure).
        "extended_reinhard" | "extended" => ToneMappingMethod::ExtendedReinhard {
            max_luma: config.max_luma,
        },

        // Reinhard + colour preservation hybrid.
        "reinhard_jodie" | "reinhardjodie" | "jodie" => ToneMappingMethod::ReinhardJodie,

        // Uncharted 2 filmic curve.
        "filmic" | "uncharted" | "uncharted2" => ToneMappingMethod::Filmic,

        // Academy Color Encoding System filmic.
        "aces" => ToneMappingMethod::Aces,

        // Simple hard clamp.
        "clamp" => ToneMappingMethod::Clamp,

        // Blender Ansel/Darktable-style filmic spline.
        "filmic_spline" | "filmicspline" | "spline" => {
            ToneMappingMethod::FilmicSpline(FilmicSplineParameters {
                output_power: config.filmic_spline.output_power,
                latitude: config.filmic_spline.latitude,
                white_point_source: config.filmic_spline.white_point_source,
                black_point_source: config.filmic_spline.black_point_source,
                contrast: config.filmic_spline.contrast,
                black_point_target: config.filmic_spline.black_point_target,
                grey_point_target: config.filmic_spline.grey_point_target,
                white_point_target: config.filmic_spline.white_point_target,
                balance: config.filmic_spline.balance,
                saturation: config.filmic_spline.saturation,
            })
        }

        // Blender AgX.
        "agx" | "ag_x" => {
            let look = build_agx_look(config)?;
            ToneMappingMethod::Agx(look)
        }

        other => {
            return Err(wm_err!(
                "unsupported tonemapping method: {} (options: itu2408, tuned_reinhard, \
                 extended_reinhard, reinhard, reinhard_jodie, filmic, aces, clamp, \
                 filmic_spline, agx)",
                other
            ));
        }
    };

    // --- 3. Build the Mapping Color Space -----------------------------------
    let gamut_clipping = match config.gamut_clipping.as_str() {
        "noclip" | "none" | "off" | "false" => GamutClipping::NoClip,
        "clip" | "soft" | "on" | "true" => GamutClipping::Clip,
        other => {
            return Err(wm_err!(
                "unsupported gamut_clipping: {} (use noclip or clip)",
                other
            ))
        }
    };

    let exposure = config.exposure;
    let content_brightness = config.content_brightness.unwrap_or(config.nits);

    let color_space_kind = match config.color_space.as_str() {
        "rgb" => ColorSpaceKind::Rgb,
        "yrg" | "default" => ColorSpaceKind::Yrg,
        "jzazbz" | "jz_azbz" | "jzczaz" => ColorSpaceKind::Jzazbz,
        other => {
            return Err(wm_err!(
                "unsupported color_space: {} (use rgb, yrg, or jzazbz)",
                other
            ))
        }
    };

    // We construct a fresh `MappingColorSpace` per call site so the type
    // doesn't need to be `Copy`.
    let make_mapping = || match color_space_kind {
        ColorSpaceKind::Rgb => MappingColorSpace::Rgb(RgbToneMapperParameters {
            gamut_clipping,
            exposure,
        }),
        ColorSpaceKind::Yrg => MappingColorSpace::Yrg(CommonToneMapperParameters {
            exposure,
            gamut_clipping,
        }),
        ColorSpaceKind::Jzazbz => MappingColorSpace::Jzazbz(JzazbzToneMapperParameters {
            content_brightness,
            exposure,
            gamut_clipping,
        }),
    };

    let map_creation_err = |e: ForgeError| wm_err!("Failed to create tone mapper: {}", e);
    let alloc_err = || wm_err!("Failed to allocate SDR output image buffer");

    // gainforge only exposes RGB/RGBA lanes, so grayscale is tone-mapped by
    // replicating L into RGB and collapsing back to luma afterwards. A neutral
    // (R=G=B) pixel stays neutral through every mapper/working space, so the
    // round-trip is lossless for the grey axis.
    let gray_kind = match &image.pixels {
        DynamicImage::ImageLuma8(_) => Some(GrayKind::L8),
        DynamicImage::ImageLumaA8(_) => Some(GrayKind::La8),
        DynamicImage::ImageLuma16(_) => Some(GrayKind::L16),
        DynamicImage::ImageLumaA16(_) => Some(GrayKind::La16),
        _ => None,
    };

    // The buffer we actually feed to the mapper: the pixels themselves for RGB
    // inputs, or an owned RGB(A) expansion for grayscale inputs.
    let src_dyn: Cow<DynamicImage> = match gray_kind {
        None => Cow::Borrowed(&image.pixels),
        Some(GrayKind::L8) => Cow::Owned(DynamicImage::ImageRgb8(image.pixels.to_rgb8())),
        Some(GrayKind::La8) => Cow::Owned(DynamicImage::ImageRgba8(image.pixels.to_rgba8())),
        Some(GrayKind::L16) => Cow::Owned(DynamicImage::ImageRgb16(image.pixels.to_rgb16())),
        Some(GrayKind::La16) => Cow::Owned(DynamicImage::ImageRgba16(image.pixels.to_rgba16())),
    };

    // --- 4. Tone-map at the native bit depth of the image -------------------
    let mapped_rgb = match &*src_dyn {
        DynamicImage::ImageRgb8(buf) => {
            let (width, height) = buf.dimensions();
            let mapper = create_tone_mapper_rgb(
                &src_profile,
                &dst_profile,
                tone_mapping_method,
                make_mapping(),
            )
            .map_err(map_creation_err)?;
            let dst = map_rows(buf.as_raw(), width as usize * 3, |s, d| {
                mapper.tonemap_lane(s, d)
            })?;
            DynamicImage::ImageRgb8(RgbImage::from_raw(width, height, dst).ok_or_else(alloc_err)?)
        }

        DynamicImage::ImageRgba8(buf) => {
            let (width, height) = buf.dimensions();
            let mapper = create_tone_mapper_rgba(
                &src_profile,
                &dst_profile,
                tone_mapping_method,
                make_mapping(),
            )
            .map_err(map_creation_err)?;
            let dst = map_rows(buf.as_raw(), width as usize * 4, |s, d| {
                mapper.tonemap_lane(s, d)
            })?;
            DynamicImage::ImageRgba8(RgbaImage::from_raw(width, height, dst).ok_or_else(alloc_err)?)
        }

        DynamicImage::ImageRgb16(buf) => {
            let (width, height) = buf.dimensions();
            let mapper = create_tone_mapper_rgb16(
                &src_profile,
                &dst_profile,
                tone_mapping_method,
                make_mapping(),
            )
            .map_err(map_creation_err)?;
            let dst = map_rows(buf.as_raw(), width as usize * 3, |s, d| {
                mapper.tonemap_lane(s, d)
            })?;
            DynamicImage::ImageRgb16(
                ImageBuffer::<ImgRgb<u16>, Vec<u16>>::from_raw(width, height, dst)
                    .ok_or_else(alloc_err)?,
            )
        }

        DynamicImage::ImageRgba16(buf) => {
            let (width, height) = buf.dimensions();
            let mapper = create_tone_mapper_rgba16(
                &src_profile,
                &dst_profile,
                tone_mapping_method,
                make_mapping(),
            )
            .map_err(map_creation_err)?;
            let dst = map_rows(buf.as_raw(), width as usize * 4, |s, d| {
                mapper.tonemap_lane(s, d)
            })?;
            DynamicImage::ImageRgba16(
                ImageBuffer::<ImgRgba<u16>, Vec<u16>>::from_raw(width, height, dst)
                    .ok_or_else(alloc_err)?,
            )
        }

        // 32-bit float. gainforge has no f32 lane (its highest-precision mapper
        // is 16-bit), so we quantize to u16 and map that. NOTE: this treats the
        // float samples as the source-encoded signal in [0, 1]; linear
        // scene-referred values above 1.0 (e.g. from EXR/Radiance) will be
        // clamped by to_rgb16/to_rgba16. If that's your case, encode to the
        // source transfer function before this step.
        DynamicImage::ImageRgb32F(_) => {
            let buf = src_dyn.to_rgb16();
            let (width, height) = buf.dimensions();
            let mapper = create_tone_mapper_rgb16(
                &src_profile,
                &dst_profile,
                tone_mapping_method,
                make_mapping(),
            )
            .map_err(map_creation_err)?;
            let dst = map_rows(buf.as_raw(), width as usize * 3, |s, d| {
                mapper.tonemap_lane(s, d)
            })?;
            DynamicImage::ImageRgb16(
                ImageBuffer::<ImgRgb<u16>, Vec<u16>>::from_raw(width, height, dst)
                    .ok_or_else(alloc_err)?,
            )
        }

        DynamicImage::ImageRgba32F(_) => {
            let buf = src_dyn.to_rgba16();
            let (width, height) = buf.dimensions();
            let mapper = create_tone_mapper_rgba16(
                &src_profile,
                &dst_profile,
                tone_mapping_method,
                make_mapping(),
            )
            .map_err(map_creation_err)?;
            let dst = map_rows(buf.as_raw(), width as usize * 4, |s, d| {
                mapper.tonemap_lane(s, d)
            })?;
            DynamicImage::ImageRgba16(
                ImageBuffer::<ImgRgba<u16>, Vec<u16>>::from_raw(width, height, dst)
                    .ok_or_else(alloc_err)?,
            )
        }

        // Catch-all for any future DynamicImage variants: convert to 16-bit first.
        // (Grayscale and float are handled explicitly above.)
        other => {
            if other.color().has_alpha() {
                let buf = other.to_rgba16();
                let (width, height) = buf.dimensions();
                let mapper = create_tone_mapper_rgba16(
                    &src_profile,
                    &dst_profile,
                    tone_mapping_method,
                    make_mapping(),
                )
                .map_err(map_creation_err)?;
                let dst = map_rows(buf.as_raw(), width as usize * 4, |s, d| {
                    mapper.tonemap_lane(s, d)
                })?;
                DynamicImage::ImageRgba16(
                    ImageBuffer::<ImgRgba<u16>, Vec<u16>>::from_raw(width, height, dst)
                        .ok_or_else(alloc_err)?,
                )
            } else {
                let buf = other.to_rgb16();
                let (width, height) = buf.dimensions();
                let mapper = create_tone_mapper_rgb16(
                    &src_profile,
                    &dst_profile,
                    tone_mapping_method,
                    make_mapping(),
                )
                .map_err(map_creation_err)?;
                let dst = map_rows(buf.as_raw(), width as usize * 3, |s, d| {
                    mapper.tonemap_lane(s, d)
                })?;
                DynamicImage::ImageRgb16(
                    ImageBuffer::<ImgRgb<u16>, Vec<u16>>::from_raw(width, height, dst)
                        .ok_or_else(alloc_err)?,
                )
            }
        }
    };

    // Collapse the tone-mapped RGB back to luma for grayscale inputs so the
    // output pixel format matches the input.
    let mapped = match gray_kind {
        None => mapped_rgb,
        Some(GrayKind::L8) => DynamicImage::ImageLuma8(mapped_rgb.to_luma8()),
        Some(GrayKind::La8) => DynamicImage::ImageLumaA8(mapped_rgb.to_luma_alpha8()),
        Some(GrayKind::L16) => DynamicImage::ImageLuma16(mapped_rgb.to_luma16()),
        Some(GrayKind::La16) => DynamicImage::ImageLumaA16(mapped_rgb.to_luma_alpha16()),
    };

    image.pixels = mapped;

    // The pixels are now sRGB. Whatever colour metadata `Image` carries (ICC
    // blob and/or cICP) must be retagged to sRGB here, otherwise a downstream
    // encoder will mis-label the output as BT.2020 PQ / HLG. sRGB cICP is
    // [1, 13, 0, 1]; an sRGB ICC profile is `ColorProfile::new_srgb()`.
    // TODO(color-metadata): reset image.<icc/cicp> to sRGB once that field is wired up.

    Ok(())
}

// -----------------------------------------------------------------------------
// AgX look construction helper
// -----------------------------------------------------------------------------

fn build_agx_look(config: &TonemapConfig) -> Result<AgxLook, MagickError> {
    let look = match config.agx_look.as_str() {
        "" | "default" => AgxLook::Agx,
        "punchy" => AgxLook::Punchy,
        "golden" => AgxLook::Golden,

        "custom" => AgxLook::Custom(AgxCustomLook {
            slope: Rgb::new(
                config.agx_custom.slope[0],
                config.agx_custom.slope[1],
                config.agx_custom.slope[2],
            ),
            power: Rgb::new(
                config.agx_custom.power[0],
                config.agx_custom.power[1],
                config.agx_custom.power[2],
            ),
            saturation: Rgb::new(
                config.agx_custom.saturation[0],
                config.agx_custom.saturation[1],
                config.agx_custom.saturation[2],
            ),
            offset: Rgb::new(
                config.agx_custom.offset[0],
                config.agx_custom.offset[1],
                config.agx_custom.offset[2],
            ),
        }),
        other => {
            return Err(wm_err!(
                "unsupported agx_look: {} (use default, punchy, golden, or custom)",
                other
            ))
        }
    };
    Ok(look)
}
