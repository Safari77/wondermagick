use siphasher::sip::SipHasher13;
use std::borrow::Cow;
use std::hash::BuildHasher;
use std::hash::Hasher;

use crate::{arg_parse_err::ArgParseErr, error::MagickError, image::Image, wm_err};
use image::{DynamicImage, RgbImage};
use rayon::prelude::*;

#[derive(Debug, Clone, PartialEq)]
pub struct QuantizeConfig {
    pub colors: u32,
    pub dither_level: f32, // 0.0 means disabled, 1.0 means full dither (clamped to 0.0..=1.0)
    pub bias: f32,         // algorithm selector, see parse_arg docs
    pub light_boost: f32, // >1.0 preserves bright detail, 1.0 is neutral (only used with Oklab k-means++)
    pub lc_priority: f32, // 0.0 = equalize lightness only, 1.0 = equalize chroma only, 0.5 = both equally
    pub brightness_preserve: f32, // 0.0 = disabled, 1.0 = fully match source mean L
}

impl Default for QuantizeConfig {
    fn default() -> Self {
        Self {
            colors: 16,
            dither_level: 0.0,
            bias: 0.0,
            light_boost: 1.0,
            lc_priority: 0.0,
            brightness_preserve: 0.0,
        }
    }
}

#[derive(Clone, Debug)]
struct CentroidsSoA {
    l: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
    chroma: Vec<f32>,
}

impl CentroidsSoA {
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            l: Vec::with_capacity(cap),
            a: Vec::with_capacity(cap),
            b: Vec::with_capacity(cap),
            chroma: Vec::with_capacity(cap),
        }
    }

    #[inline(always)]
    pub fn push(&mut self, pt: OkPt) {
        self.l.push(pt.l);
        self.a.push(pt.a);
        self.b.push(pt.b);
        self.chroma.push(pt.chroma);
    }

    #[inline(always)]
    pub fn update(&mut self, idx: usize, pt: OkPt) {
        self.l[idx] = pt.l;
        self.a[idx] = pt.a;
        self.b[idx] = pt.b;
        self.chroma[idx] = pt.chroma;
    }

    #[inline(always)]
    pub fn get_pt(&self, idx: usize) -> OkPt {
        OkPt { l: self.l[idx], a: self.a[idx], b: self.b[idx], chroma: self.chroma[idx] }
    }

    pub fn to_oklab_vec(&self) -> Vec<Oklab> {
        (0..self.l.len()).map(|i| Oklab { l: self.l[i], a: self.a[i], b: self.b[i] }).collect()
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.l.len()
    }
}

#[inline(always)]
fn oklch_weighted_dist_soa_single(p: OkPt, soa: &CentroidsSoA, idx: usize) -> f32 {
    let dl = p.l - soa.l[idx];
    let dc = p.chroma - soa.chroma[idx];
    let da = p.a - soa.a[idx];
    let db = p.b - soa.b[idx];
    let dh_sq = (da * da + db * db - dc * dc).max(0.0);
    // Perceptually balanced weights: 1.0 Lightness, 2.0 Chroma, 3.5 Hue.
    // Balances contrast preservation in light/dark areas while ensuring distinct
    // hue tints (e.g. subtle blues in dark/gray scenes) are not collapsed into gray.
    (dl * dl * 1.0) + (dc * dc * 2.0) + (dh_sq * 3.5)
}

/// Parse an f32 that must be finite — NaN or ±inf in any of these parameters
/// would silently poison palette weights/distances downstream.
fn parse_finite_f32(s: &str, err: &'static str) -> Result<f32, ArgParseErr> {
    match s.trim().parse::<f32>() {
        Ok(v) if v.is_finite() => Ok(v),
        _ => Err(ArgParseErr::with_msg(err)),
    }
}

// Draw domains
const TAG_FIRST: u8 = 0; // first-centroid weighted threshold
const TAG_FALLBACK: u8 = 1; // degenerate-case random index for centroid ki
const TAG_TARGET_BASE: u8 = 10; // weighted-pick target base for candidate trials

/// Two-tier SipHash13 RNG.
///
/// Init: metadata (24 B) is hashed ONCE under a fixed master key; the 64-bit
/// result, zero-padded to 16 bytes, becomes the key for all draws.
/// Draw: SipHash13(derived_key, tag || index) — only 9 bytes per call.
///
/// Stateless (no counter): every draw is a pure function of (tag, index),
/// so `&self` suffices and call order doesn't matter.
struct SipRng {
    key: [u8; 16],
}

impl SipRng {
    fn new(k: usize, n_unique: usize, n_pixels: usize) -> Self {
        const MASTER_KEY: &[u8; 16] = b"palette-mstr-key";

        // Structural Metadata Mix, little-endian → identical bytes everywhere.
        let mut metadata = [0u8; 32];
        let nonce: u64 = std::collections::hash_map::RandomState::new().build_hasher().finish();
        metadata[0..8].copy_from_slice(&(k as u64).to_le_bytes());
        metadata[8..16].copy_from_slice(&(n_unique as u64).to_le_bytes());
        metadata[16..24].copy_from_slice(&(n_pixels as u64).to_le_bytes());
        metadata[24..32].copy_from_slice(&nonce.to_le_bytes());
        // One-time metadata mix → 64-bit value.
        let mut h = SipHasher13::new_with_key(MASTER_KEY);
        h.write(&metadata);
        let mix = h.finish();

        // Pad to the 16-byte key SipHasher requires.
        let mut key = [0u8; 16];
        key[..8].copy_from_slice(&mix.to_le_bytes());
        Self { key }
    }

    #[inline]
    fn draw(&self, tag: u8, i: u64) -> u64 {
        let mut h = SipHasher13::new_with_key(&self.key);
        h.write_u8(tag);
        h.write_u64(i);
        h.finish()
    }

    /// Uniform in [0, 1) — top 24 bits, exact 2^-24 scaling onto the f32 mantissa.
    #[inline]
    fn unit_f32(&self, tag: u8, i: u64) -> f32 {
        const SCALE: f32 = 1.0 / (1u64 << 24) as f32;
        (self.draw(tag, i) >> 40) as f32 * SCALE
    }

    /// Index in [0, n) — modulo BEFORE the usize cast (32-bit-safe).
    #[inline]
    fn below(&self, tag: u8, i: u64, n: usize) -> usize {
        (self.draw(tag, i) % n as u64) as usize
    }
}

impl QuantizeConfig {
    /// Parse from a comma-separated string of three values, or "default".
    /// Format: "colors,dither_level,bias[:light_boost[:lc_priority[:brightness_preserve]]]"
    ///
    ///    colors:  palette size (2-)
    ///    dither:  error diffusion strength (0.0 = off, 1.0 = full)
    ///    bias:    algorithm selector:
    ///               <= -2.0        MacQueen online k-means in Oklab (stochastic, fast)
    ///          -2.0 < ... < 0.0    Oklab median-cut (fast, good for flat art)
    ///                    0.0        classic RGB k-means (mapping & dither run in RGB too)
    ///                  > 0.0        Oklab k-means++ (perceptual, value = saturation boost)
    ///
    ///  Oklab k-means++ accepts optional suffixes after bias:
    ///    light_boost      highlight preservation (default 1.0, higher keeps brights)
    ///    light_boost:lc   equalization blend (0.0 = lightness, 1.0 = chroma, default 0.0)
    ///    light_boost:lc:bright brightness preservation (0.0 = off, 1.0 = match source mean L, default 0.0)
    ///
    ///  Examples:  16,1.0,0.0          16 colors, full dither, RGB k-means
    ///             8,0.5,-1.0            8 colors, half dither, Oklab median-cut
    ///             32,0.1,1.0:1.5:0.5  32 colors, Oklab K-means with chroma+lightness equalization balanced
    pub fn parse_arg(s: &str) -> Result<Self, ArgParseErr> {
        let s = s.trim();
        if s.eq_ignore_ascii_case("default") {
            return Ok(Self::default());
        }

        // Iterator-based splitting: no allocations, everything borrows from `s`.
        let mut parts = s.split(',');
        let (colors_str, dither_str, bias_str) = match (
            parts.next(),
            parts.next(),
            parts.next(),
            parts.next(),
        ) {
            (Some(c), Some(d), Some(b), None) => (c, d, b),
            _ => {
                return Err(ArgParseErr::with_msg(
                    "quantize requires 'default' or exactly 3 comma-separated values: \
                         colors,dither_level,bias (bias may include :light_boost:lc_priority:brightness_preserve, \
                         e.g. 1.0:5.5:0.5:0.8)",
                ));
            }
        };

        let colors = colors_str.trim().parse::<u32>().map_err(|_| {
            ArgParseErr::with_msg("invalid colors value (must be positive integer)")
        })?;

        let dither_level =
            parse_finite_f32(dither_str, "invalid dither_level value (must be a finite float)")?
                .clamp(0.0, 1.0);

        // bias field supports optional :light_boost, :lc_priority and :brightness_preserve suffixes
        // (e.g. "1.0:5.5" or "1.0:5.5:0.5" or "1.0:5.5:0.5:0.8")
        let mut bias_parts = bias_str.trim().split(':');
        let bias = parse_finite_f32(
            bias_parts.next().unwrap_or(""),
            "invalid bias value (must be a finite float)",
        )?;
        let light_boost = match bias_parts.next() {
            Some(v) => {
                let x = parse_finite_f32(v, "invalid light_boost value (must be a finite float)")?;
                if x <= 0.0 {
                    return Err(ArgParseErr::with_msg("light_boost must be > 0.0"));
                }
                x
            }
            None => 1.0,
        };
        let lc_priority = match bias_parts.next() {
            Some(v) => {
                parse_finite_f32(v, "invalid lc_priority value (must be a finite float 0.0-1.0)")?
                    .clamp(0.0, 1.0)
            }
            None => 0.0,
        };
        let brightness_preserve = match bias_parts.next() {
            Some(v) => parse_finite_f32(
                v,
                "invalid brightness_preserve value (must be a finite float 0.0-1.0)",
            )?
            .clamp(0.0, 1.0),
            None => 0.0,
        };

        if bias_parts.next().is_some() {
            return Err(ArgParseErr::with_msg(
                "bias accepts at most three ':' suffixes: bias:light_boost:lc_priority:brightness_preserve",
            ));
        }

        Ok(Self { colors, dither_level, bias, light_boost, lc_priority, brightness_preserve })
    }
}

/// Borrow the pixels as RGB8 without copying when the image already is RGB8;
/// otherwise convert once into an owned buffer. The explicit lifetime ties the
/// borrowed variant of the returned `Cow` to the source image — no `to_rgb8()`
/// copy happens on the (common) already-RGB path.
fn as_rgb8<'a>(img: &'a DynamicImage) -> Cow<'a, RgbImage> {
    match img.as_rgb8() {
        Some(rgb) => Cow::Borrowed(rgb),
        None => Cow::Owned(img.to_rgb8()),
    }
}

/// Per-channel error clamp applied at READ in the dither path: prevents
/// unbounded error accumulation that bleeds large one-color blobs across
/// regions. RGB clamps are ~the same relative magnitude as the Oklab ones
/// (0.2 * 255 * 0.94 ≈ 48).
const ERR_CLAMP_OKLAB: [f32; 3] = [0.2, 0.05, 0.05];
const ERR_CLAMP_RGB: [f32; 3] = [48.0; 3];

pub fn quantize(image: &mut Image, config: &QuantizeConfig) -> Result<(), MagickError> {
    if config.colors < 2 {
        return Err(wm_err!("quantize requires at least 2 colors"));
    }
    let width = image.pixels.width() as usize;
    let height = image.pixels.height() as usize;
    if width == 0 || height == 0 {
        return Err(wm_err!("quantize requires a non-empty image"));
    }

    // Everything in this block only borrows `image.pixels`; the replacement
    // image is assigned after the block, so the borrow ends before mutation.
    let out_buf: Vec<u8> = {
        let input = as_rgb8(&image.pixels);

        // Zero-copy view of the tightly packed RGBRGB… buffer as [[u8; 3]].
        // Requires Rust >= 1.88; on older toolchains fall back to
        // `input.pixels().map(|p| p.0).collect::<Vec<_>>()`.
        let (pixels, remainder) = input.as_raw().as_chunks::<3>();
        debug_assert!(remainder.is_empty());
        let pixels: &[[u8; 3]] = pixels;

        let k = config.colors as usize;

        // 1. Generate the palette. `bias == 0.0` is the "classic" RGB mode:
        //    palette, mapping AND dithering all run in RGB so the selection
        //    metric is consistent end-to-end. Every other mode works in Oklab.
        let rgb_space = config.bias == 0.0;
        let palette = if config.bias <= -2.0 {
            generate_palette_macqueen(pixels, k, width, config.dither_level)
        } else if config.bias < 0.0 {
            generate_palette_median_cut(pixels, k)
        } else if config.bias > 0.0 {
            generate_palette_oklab(
                pixels,
                k,
                config.bias,
                config.light_boost,
                config.lc_priority,
                config.brightness_preserve,
            )
        } else {
            generate_palette_rgb(pixels, k)
        };

        if palette.is_empty() {
            return Err(wm_err!("quantize failed to generate a color palette"));
        }

        // 2 & 3. Map pixels to the nearest palette color.
        //
        // STATIC dispatch into monomorphized mapping code: each branch gets a
        // copy specialized for its metric, so the N×k distance calls in the
        // inner loop inline. (Passing the metric through a `fn` pointer here
        // defeats inlining and benchmarked as a significant regression.)
        if config.dither_level > 0.0 {
            if rgb_space {
                dither_and_map(
                    pixels,
                    &palette,
                    rgb_to_f32,
                    rgb_dist_sq,
                    |x: [f32; 3]| x,
                    ERR_CLAMP_RGB,
                    [1.0; 3],
                    config.dither_level * 25.5,
                    width,
                    height,
                )
            } else {
                dither_and_map(
                    pixels,
                    &palette,
                    srgb_to_oklab_arr,
                    oklch_weighted_dist_pt,
                    OkPt::from_arr,
                    ERR_CLAMP_OKLAB,
                    [1.0, 0.0, 0.0],
                    config.dither_level * 0.1,
                    width,
                    height,
                )
            }
        } else if rgb_space {
            parallel_map(pixels, &palette, rgb_to_f32, rgb_dist_sq, |x: [f32; 3]| x)
        } else {
            parallel_map(
                pixels,
                &palette,
                srgb_to_oklab_arr,
                oklch_weighted_dist_pt,
                OkPt::from_arr,
            )
        }
    };

    let output = RgbImage::from_raw(width as u32, height as u32, out_buf)
        .expect("output buffer size matches dimensions");
    image.pixels = DynamicImage::ImageRgb8(output);
    Ok(())
}

/// Nearest palette index under `dist`. The scalar `<` comparison preserves
/// first-index tie-breaking, identical to any per-pixel loop.
///
/// Generic over the working-space point type AND the distance metric: `dist`
/// is passed as a closure/fn item (zero-sized, statically known), so the
/// N×k calls in the inner loop fully inline — never take this through a
/// runtime `fn` pointer.
#[inline(always)]
fn nearest_idx<T, D>(p: T, palette: &[T], dist: D) -> usize
where
    T: Copy,
    D: Fn(T, T) -> f32,
{
    let mut min_dist = f32::MAX;
    let mut best = 0;
    for (i, &c) in palette.iter().enumerate() {
        let d = dist(p, c);
        if d < min_dist {
            min_dist = d;
            best = i;
        }
    }
    best
}

/// Convert to working space, then run the serial hybrid blue-noise + Sierra
/// Lite error diffusion. Each instantiation statically knows `convert`,
/// `dist` and `make_pt`, so the per-pixel, per-palette-entry distance calls
/// inline.
#[allow(clippy::too_many_arguments)]
fn dither_and_map<T, C, D, M>(
    pixels: &[[u8; 3]],
    palette: &[[u8; 3]],
    convert: C,
    dist: D,
    make_pt: M,
    clamp: [f32; 3],
    jitter_mask: [f32; 3],
    noise_spread: f32,
    width: usize,
    height: usize,
) -> Vec<u8>
where
    T: Copy,
    C: Fn([u8; 3]) -> [f32; 3] + Sync,
    D: Fn(T, T) -> f32,
    M: Fn([f32; 3]) -> T,
{
    let working: Vec<[f32; 3]> = pixels.par_iter().map(|&p| convert(p)).collect();
    let palette_ws: Vec<[f32; 3]> = palette.iter().map(|&c| convert(c)).collect();
    // Chroma (and any other loop-invariant point data) computed once per
    // palette entry — k sqrts total, instead of k per pixel.
    let palette_pts: Vec<T> = palette_ws.iter().map(|&c| make_pt(c)).collect();
    let mut out = vec![0u8; width * height * 3];
    error_diffusion_map(
        &working,
        palette,
        &palette_ws,
        &palette_pts,
        &dist,
        &make_pt,
        clamp,
        jitter_mask,
        noise_spread,
        width,
        height,
        &mut out,
    );
    out
}

/// No-dither mapping, memoized per unique color: the conversion and the O(k)
/// nearest search run once per distinct color instead of once per pixel —
/// often a 10–100× reduction. Deterministic and bit-identical to per-pixel
/// mapping (the pipeline is a pure function of the RGB triple; first-index
/// tie-breaking is preserved).
fn parallel_map<T, C, D, M>(
    pixels: &[[u8; 3]],
    palette: &[[u8; 3]],
    convert: C,
    dist: D,
    make_pt: M,
) -> Vec<u8>
where
    T: Copy + Sync,
    C: Fn([u8; 3]) -> [f32; 3] + Sync,
    D: Fn(T, T) -> f32 + Sync,
    M: Fn([f32; 3]) -> T + Sync,
{
    // Remap pixel indices to unique color indices (counts not needed)
    let (unique, pixel_to_unique) = unique_colors_with_map(pixels);

    let palette_pts: Vec<T> = palette.iter().map(|&c| make_pt(convert(c))).collect();

    let best_per_unique: Vec<u32> = unique
        .par_iter()
        .map(|&c| nearest_idx(make_pt(convert(c)), &palette_pts, &dist) as u32)
        .collect();

    // Pre-map each unique color index directly to its final RGB palette entry,
    // eliminating double-array indirection (`palette[best_per_unique[...]]`)
    // inside the multi-million iteration output loop.
    let unique_to_rgb: Vec<[u8; 3]> =
        best_per_unique.iter().map(|&idx| palette[idx as usize]).collect();

    let mut out = vec![0u8; pixels.len() * 3];
    out.par_chunks_exact_mut(3).enumerate().for_each(|(i, px)| {
        px.copy_from_slice(&unique_to_rgb[pixel_to_unique[i] as usize]);
    });
    out
}

/// Serpentine Sierra Lite error diffusion with blue-noise jitter, generic over
/// the working space (Oklab or u8-scale RGB) and its point representation, so
/// both modes share one scan implementation.
///
/// `palette_ws` is the raw working-space palette (for error arithmetic);
/// `palette_pts` is the same palette with loop-invariant metric data (OkLCh
/// chroma) precomputed (for distance tests). `make_pt` builds a point from a
/// working-space pixel — for Oklab that computes the chroma sqrt ONCE per
/// pixel instead of once per (pixel × palette entry).
#[allow(clippy::too_many_arguments)]
fn error_diffusion_map<T, D, M>(
    working: &[[f32; 3]],
    palette: &[[u8; 3]],
    palette_ws: &[[f32; 3]],
    palette_pts: &[T],
    dist: &D,
    make_pt: &M,
    clamp: [f32; 3],
    jitter_mask: [f32; 3],
    noise_spread: f32,
    width: usize,
    height: usize,
    out: &mut [u8],
) where
    T: Copy,
    D: Fn(T, T) -> f32,
    M: Fn([f32; 3]) -> T,
{
    let padded_w = width + 2;
    let mut errors = vec![[0.0f32; 3]; padded_w * height];

    for y in 0..height {
        // Serpentine: even rows scan left→right, odd rows right→left.
        let ltr = y % 2 == 0;
        let fwd_dx: isize = if ltr { 1 } else { -1 };
        let diag_dx: isize = -fwd_dx;

        for step in 0..width {
            let xs = if ltr { step } else { width - 1 - step };
            let i = y * width + xs;
            let idx = y * padded_w + xs + 1;

            // Current pixel + accumulated diffused error (clamped).
            let p = working[i];
            let e = [
                errors[idx][0].clamp(-clamp[0], clamp[0]),
                errors[idx][1].clamp(-clamp[1], clamp[1]),
                errors[idx][2].clamp(-clamp[2], clamp[2]),
            ];

            // Blue-noise jitter — shifts decision boundaries between palette
            // colors, breaking up the structured patterns of pure error
            // diffusion.
            let noise = (get_noise(xs as u32, y as u32) as f32 / 255.0) - 0.5;
            let j = noise * noise_spread;

            // One point construction per pixel: for Oklab this is where the
            // single per-pixel chroma sqrt happens.
            let jp = make_pt([
                p[0] + e[0] + j * jitter_mask[0],
                p[1] + e[1] + j * jitter_mask[1],
                p[2] + e[2] + j * jitter_mask[2],
            ]);

            let best = nearest_idx(jp, palette_pts, dist);
            out[i * 3..i * 3 + 3].copy_from_slice(&palette[best]);

            // Quantization error EXCLUDES the jitter term: jitter's role is to
            // perturb which palette entry wins, but propagating it into the
            // error buffer turns that buffer into a bounded random walk that
            // eventually lands near a distant palette entry for a single pixel
            // before correction propagates.
            let chosen = palette_ws[best];
            let err =
                [(p[0] + e[0]) - chosen[0], (p[1] + e[1]) - chosen[1], (p[2] + e[2]) - chosen[2]];

            // Sierra Lite distribution (same as monochrome.rs):
            //    current → [fwd: 2/4, diag-below: 1/4, below: 1/4]
            // mirrored on R→L rows so error always follows the scan direction.
            // Edge pixels write into the padding column (never read), so no
            // explicit x-bounds check is needed.
            let fwd_col = (xs as isize + fwd_dx + 1) as usize;
            let fwd_idx = y * padded_w + fwd_col;
            errors[fwd_idx][0] += err[0] * 0.5;
            errors[fwd_idx][1] += err[1] * 0.5;
            errors[fwd_idx][2] += err[2] * 0.5;

            if y + 1 < height {
                let below_row = (y + 1) * padded_w;
                let diag_col = (xs as isize + diag_dx + 1) as usize;
                let diag_idx = below_row + diag_col;
                let below_idx = below_row + xs + 1;
                for ch in 0..3 {
                    errors[diag_idx][ch] += err[ch] * 0.25;
                    errors[below_idx][ch] += err[ch] * 0.25;
                }
            }
        }
    }
}

/// Deduplicate pixels into unique colors plus a per-pixel index map.
/// Returns `(unique_rgb, pixel_to_unique)`.
///
/// FP determinism: duplicates share identical converted values, and downstream
/// passes that must match a per-pixel FP sequence walk the ORIGINAL pixel
/// order through `pixel_to_unique` term-for-term.
fn unique_colors_with_map(pixels: &[[u8; 3]]) -> (Vec<[u8; 3]>, Vec<u32>) {
    let mut indexed_keys: Vec<(u32, u32)> = pixels
        .par_iter()
        .enumerate()
        .map(|(i, p)| {
            let key = (u32::from(p[0]) << 16) | (u32::from(p[1]) << 8) | u32::from(p[2]);
            (key, i as u32)
        })
        .collect();
    indexed_keys.par_sort_unstable_by_key(|&(k, _)| k);

    let mut unique_rgb = Vec::with_capacity(pixels.len() / 4 + 1);
    let mut pixel_to_unique = vec![0u32; pixels.len()];

    let mut i = 0;
    while i < indexed_keys.len() {
        let key = indexed_keys[i].0;
        let u_idx = unique_rgb.len() as u32;
        unique_rgb.push([(key >> 16) as u8, (key >> 8) as u8, key as u8]);

        while i < indexed_keys.len() && indexed_keys[i].0 == key {
            pixel_to_unique[indexed_keys[i].1 as usize] = u_idx;
            i += 1;
        }
    }
    (unique_rgb, pixel_to_unique)
}

/// Deduplicate pixels into unique colors and their frequency counts for histogram clustering.
/// Returns `(unique_rgb, unique_counts)`.
///
/// Stores only 32-bit color keys without pixel indices, halving memory and sort bandwidth.
fn unique_colors_with_counts(pixels: &[[u8; 3]]) -> (Vec<[u8; 3]>, Vec<u32>) {
    let mut keys: Vec<u32> = pixels
        .par_iter()
        .map(|p| (u32::from(p[0]) << 16) | (u32::from(p[1]) << 8) | u32::from(p[2]))
        .collect();
    keys.par_sort_unstable();

    let mut unique_rgb = Vec::with_capacity(pixels.len() / 4 + 1);
    let mut unique_counts = Vec::with_capacity(pixels.len() / 4 + 1);

    let mut i = 0;
    while i < keys.len() {
        let key = keys[i];
        unique_rgb.push([(key >> 16) as u8, (key >> 8) as u8, key as u8]);

        let start = i;
        while i < keys.len() && keys[i] == key {
            i += 1;
        }
        unique_counts.push((i - start) as u32);
    }
    (unique_rgb, unique_counts)
}

/// MacQueen K-Means clustering natively in Oklab perceptual space (bias <= -2.0).
/// Stochastic online updates allow rapid convergence and escape from local minima.
/// Includes blue noise jitter integration during the generation assignments to
/// gently break up uniform structural biases.
fn generate_palette_macqueen(
    rgb_pixels: &[[u8; 3]],
    k: usize,
    width: usize,
    dither_level: f32,
) -> Vec<[u8; 3]> {
    let k = k.max(1);
    let n_pixels = rgb_pixels.len();
    if n_pixels == 0 {
        return vec![[0, 0, 0]; k];
    }

    let oklab_pixels: Vec<Oklab> = rgb_pixels.par_iter().map(|&p| srgb_to_oklab(p)).collect();

    // Initial centroids: evenly spaced distinct pixel positions. (Random init
    // could — and for small palettes often did — pick the same color multiple
    // times, permanently wasting palette slots.)
    let init_step = (n_pixels / k).max(1);
    let mut centroids = vec![Oklab { l: 0.0, a: 0.0, b: 0.0 }; k];
    for (i, c) in centroids.iter_mut().enumerate() {
        *c = oklab_pixels[(i * init_step).min(n_pixels - 1)];
    }

    // Thread-local compatible fast xorshift PRNG
    let mut rng_state: u64 = 0x1234_BEEF;
    let mut xorshift = || -> u64 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        rng_state
    };

    let mut counts = vec![0u32; k];
    const BATCH_SIZE: usize = 4096;
    let samples = n_pixels.min(512 * 512); // Sample cap to prevent infinite loops on massive inputs
    // At least one iteration even for images smaller than a single batch —
    // otherwise the palette would stay at its raw initial centroids.
    let iterations = (samples / BATCH_SIZE).max(1);
    let noise_spread = dither_level * 0.1; // Reduced multiplier due to tight Oklab ranges

    let jitter = |p: Oklab, idx: usize| -> OkPt {
        if noise_spread <= 0.0 {
            return OkPt::from_oklab(p);
        }
        let noise = (get_noise((idx % width) as u32, (idx / width) as u32) as f32 / 255.0) - 0.5;
        let j = noise * noise_spread;
        OkPt::from_arr([(p.l + j).clamp(0.0, 1.0), p.a + j, p.b + j])
    };

    // MacQueen iterations
    for _ in 0..iterations {
        let batch_indices: Vec<usize> =
            (0..BATCH_SIZE).map(|_| (xorshift() as usize) % n_pixels).collect();

        // Centroids are fixed during the assignment phase of a batch —
        // k chroma sqrts here instead of k per sample.
        let centroid_pts: Vec<OkPt> = centroids.iter().map(|&c| OkPt::from_oklab(c)).collect();

        // Step 1: in parallel, sample colors and find their nearest current
        // centroid. Uses the same oklch_weighted_dist as the mapping pass so
        // centroids are optimized under the selection metric. The jittered
        // pixel (with its chroma) is computed once and reused for the update.
        let assignments: Vec<(usize, OkPt)> = batch_indices
            .par_iter()
            .map(|&idx| {
                let p = jitter(oklab_pixels[idx], idx);
                (nearest_idx(p, &centroid_pts, oklch_weighted_dist_pt), p)
            })
            .collect();

        // Step 2: sequentially update the centroids based on assignments
        for (c_idx, p) in assignments {
            counts[c_idx] += 1;
            // Adaptive learning rate: 1.0 / sqrt(count)
            let rate = 1.0 / (counts[c_idx] as f32).sqrt();
            let c = &mut centroids[c_idx];
            c.l += rate * (p.l - c.l);
            c.a += rate * (p.a - c.a);
            c.b += rate * (p.b - c.b);
        }
    }

    // Final mapping back to sRGB [u8; 3]
    centroids.into_iter().map(oklab_to_srgb_u8).collect()
}

/// Original Fast RGB K-Means (used when bias == 0.0)
fn generate_palette_rgb(pixels: &[[u8; 3]], k: usize) -> Vec<[u8; 3]> {
    let k = k.max(1);
    if pixels.is_empty() {
        return vec![[0, 0, 0]; k];
    }

    let mut centroids = vec![[0.0f32; 3]; k];
    let step = (pixels.len() / k).max(1);
    for (i, c) in centroids.iter_mut().enumerate() {
        let p = pixels[(i * step).min(pixels.len() - 1)];
        *c = [p[0] as f32, p[1] as f32, p[2] as f32];
    }

    let max_iterations = 15;
    for _ in 0..max_iterations {
        let (new_sums, new_counts) = pixels
            .par_iter()
            .fold(
                || (vec![0.0f32; 3 * k], vec![0usize; k]),
                |mut acc: (Vec<f32>, Vec<usize>), p| {
                    let pf = rgb_to_f32(*p);
                    let best = nearest_idx(pf, &centroids, rgb_dist_sq);
                    acc.0[best * 3] += pf[0];
                    acc.0[best * 3 + 1] += pf[1];
                    acc.0[best * 3 + 2] += pf[2];
                    acc.1[best] += 1;
                    acc
                },
            )
            .reduce(
                || (vec![0.0f32; 3 * k], vec![0usize; k]),
                |mut a, b| {
                    for i in 0..3 * k {
                        a.0[i] = a.0[i].algebraic_add(b.0[i]);
                    }
                    for i in 0..k {
                        a.1[i] += b.1[i];
                    }
                    a
                },
            );

        let mut changed = false;
        for (i, c) in centroids.iter_mut().enumerate() {
            if new_counts[i] > 0 {
                let count = new_counts[i] as f32;
                let new_c = [
                    new_sums[i * 3] / count,
                    new_sums[i * 3 + 1] / count,
                    new_sums[i * 3 + 2] / count,
                ];
                if (c[0] - new_c[0]).abs() > 0.5
                    || (c[1] - new_c[1]).abs() > 0.5
                    || (c[2] - new_c[2]).abs() > 0.5
                {
                    changed = true;
                }
                *c = new_c;
            }
        }
        if !changed {
            break;
        }
    }

    centroids
        .iter()
        .map(|c| {
            [
                c[0].clamp(0.0, 255.0).round() as u8,
                c[1].clamp(0.0, 255.0).round() as u8,
                c[2].clamp(0.0, 255.0).round() as u8,
            ]
        })
        .collect()
}

/// Median-cut palette generation in Oklab perceptual space (used when
/// -2.0 < bias < 0.0). Recursively splits the color space by the channel with
/// highest variance, which naturally preserves dark tones and minority color
/// clusters.
fn generate_palette_median_cut(rgb_pixels: &[[u8; 3]], k: usize) -> Vec<[u8; 3]> {
    let k = k.max(1);
    let mut oklab_pixels: Vec<Oklab> = rgb_pixels.par_iter().map(|&p| srgb_to_oklab(p)).collect();

    if oklab_pixels.is_empty() {
        return vec![[0, 0, 0]; k];
    }

    // Each box is a contiguous range in the pixel array. After partitioning a
    // box around the median of its highest-variance channel, the two child
    // boxes are themselves contiguous — no copies needed.
    struct McBox {
        start: usize,
        len: usize,
        /// Which Oklab channel (0=L, 1=a, 2=b) has the highest weighted variance
        split_channel: usize,
        /// Total weighted variance across all channels — used as the priority
        /// key. Because variance is computed as a SUM of squared deviations
        /// (not an average), it naturally scales with pixel count, so large
        /// varied boxes are split first.
        priority: f32,
    }

    // Oklab's a/b channels have a much narrower natural range (~±0.3) than L (0–1).
    // Without scaling, L dominates all splits and chromatic differences (skin hues,
    // blue vs. gray) are ignored. These weights equalize the channels' contribution
    // to variance so that hue/chroma splits happen when they should.
    const CHANNEL_WEIGHT: [f32; 3] = [1.0, 3.0, 3.0];

    let compute_box_stats = |pixels: &[Oklab], start: usize, len: usize| -> McBox {
        let slice = &pixels[start..start + len];
        let n = len as f32;

        let mut sum = [0.0f32; 3];
        for p in slice {
            sum[0] = sum[0].algebraic_add(p.l);
            sum[1] = sum[1].algebraic_add(p.a);
            sum[2] = sum[2].algebraic_add(p.b);
        }
        let mean = [sum[0] / n, sum[1] / n, sum[2] / n];

        let mut var = [0.0f32; 3];
        for p in slice {
            let vals = [p.l, p.a, p.b];
            for ch in 0..3 {
                let d = vals[ch] - mean[ch];
                var[ch] += d * d;
            }
        }
        let weighted_var =
            [var[0] * CHANNEL_WEIGHT[0], var[1] * CHANNEL_WEIGHT[1], var[2] * CHANNEL_WEIGHT[2]];

        let (split_channel, _) = weighted_var
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        McBox {
            start,
            len,
            split_channel,
            priority: weighted_var[0] + weighted_var[1] + weighted_var[2],
        }
    };

    // Start with one box containing all pixels
    let mut boxes: Vec<McBox> = vec![compute_box_stats(&oklab_pixels, 0, oklab_pixels.len())];

    while boxes.len() < k {
        // Find the box with the highest total weighted variance to split
        let best_idx = boxes
            .iter()
            .enumerate()
            .filter(|(_, b)| b.len >= 2)
            .max_by(|a, b| {
                a.1.priority.partial_cmp(&b.1.priority).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i);

        let best_idx = match best_idx {
            Some(i) => i,
            None => break, // all boxes have 1 pixel, can't split further
        };

        let bx = boxes.swap_remove(best_idx);
        let mid = bx.len / 2;

        // Partition around the median in O(n) instead of a full O(n log n)
        // sort — median-cut only needs the two halves, not order within them.
        // With tied values both halves contain the same multiset of values as
        // a sorted split would, so box statistics are identical.
        let slice = &mut oklab_pixels[bx.start..bx.start + bx.len];
        match bx.split_channel {
            0 => slice.select_nth_unstable_by(mid, |a, b| a.l.total_cmp(&b.l)),
            1 => slice.select_nth_unstable_by(mid, |a, b| a.a.total_cmp(&b.a)),
            _ => slice.select_nth_unstable_by(mid, |a, b| a.b.total_cmp(&b.b)),
        };

        let left = compute_box_stats(&oklab_pixels, bx.start, mid);
        let right = compute_box_stats(&oklab_pixels, bx.start + mid, bx.len - mid);

        boxes.push(left);
        boxes.push(right);
    }

    // Average each box in parallel to produce the final palette
    boxes
        .par_iter()
        .map(|bx| {
            let slice = &oklab_pixels[bx.start..bx.start + bx.len];
            let n = slice.len() as f64;
            let (sum_l, sum_a, sum_b) =
                slice.iter().fold((0.0f64, 0.0f64, 0.0f64), |(sl, sa, sb), p| {
                    (
                        sl.algebraic_add(p.l as f64),
                        sa.algebraic_add(p.a as f64),
                        sb.algebraic_add(p.b as f64),
                    )
                });
            oklab_to_srgb_u8(Oklab {
                l: (sum_l / n) as f32,
                a: (sum_a / n) as f32,
                b: (sum_b / n) as f32,
            })
        })
        .collect()
}

/// Post-palette luminance normalization that preserves relative lightness
/// spacing. Instead of shifting all entries by a uniform delta (which clamps
/// highlights and collapses bright detail), this applies a power (gamma) curve
/// in L space anchored at L=0 and L=1. This maintains highlight separation
/// unconditionally without clipping any bright detail.
///
/// The correction is blended between the normalized palette and the original
/// using `strength`, where 0.0 = no correction and 1.0 = full match to
/// source mean L.
fn preserve_brightness(palette_oklab: &mut [Oklab], rgb_pixels: &[[u8; 3]], strength: f32) {
    if strength <= 0.0 || palette_oklab.is_empty() || rgb_pixels.is_empty() {
        return;
    }

    // Source mean L (sampled for large images)
    let sample_len = rgb_pixels.len().min(256 * 256);
    let step = (rgb_pixels.len() / sample_len).max(1);
    let mut source_l_sum = 0.0f64;
    let mut source_count = 0u64;
    for p in rgb_pixels.iter().step_by(step) {
        source_l_sum += srgb_to_oklab(*p).l as f64;
        source_count += 1;
    }
    if source_count == 0 {
        return;
    }
    let source_mean_l = (source_l_sum / source_count as f64) as f32;

    // Palette mean L
    let palette_mean_l: f32 =
        palette_oklab.iter().map(|c| c.l as f64).sum::<f64>() as f32 / palette_oklab.len() as f32;

    if palette_mean_l <= 1e-4 || source_mean_l <= 1e-4 {
        return;
    }

    // Power curve mapping: L -> L^gamma. Preserves relative spacing while avoiding
    // highlight clamping. Clamped to avoid extreme distortion.
    let target_mean = palette_mean_l + (source_mean_l - palette_mean_l) * strength;
    let gamma = (target_mean.ln() / palette_mean_l.ln()).clamp(0.5, 2.0);

    for c in palette_oklab.iter_mut() {
        c.l = c.l.powf(gamma).clamp(0.0, 1.0);
    }
}

/// K-means++ clustering in Oklab with the weighted OkLCh distance (used when bias > 0.0).
///
/// 3D histogram equalization across (L, a, b) space ensures dark areas, minority colors
/// (e.g. subtle blues in dark/gray scenes), and delicate highlights receive dedicated
/// representation in the final palette. sat_bias boosts saturated pixels' weight,
/// light_boost controls highlight preservation, and lc_priority balances lightness
/// and chromatic discrimination.
fn generate_palette_oklab(
    rgb_pixels: &[[u8; 3]],
    k: usize,
    sat_bias: f32,
    light_boost: f32,
    lc_priority: f32,
    brightness_preserve: f32,
) -> Vec<[u8; 3]> {
    let k = k.max(1);

    if rgb_pixels.is_empty() {
        return vec![[0, 0, 0]; k];
    }

    // ── Color histogram deduplication ───────────────────────────────────────
    let (unique_rgb, unique_counts) = unique_colors_with_counts(rgb_pixels);
    let n_unique = unique_rgb.len();
    let k = k.min(n_unique).max(1);

    let oklab_pixels: Vec<Oklab> = unique_rgb.par_iter().map(|&p| srgb_to_oklab(p)).collect();
    let oklab_pts: Vec<OkPt> = oklab_pixels.par_iter().map(|&o| OkPt::from_oklab(o)).collect();

    // 1. Per-pixel weights: uniform base with optional linear chroma boost and highlight boost.
    let chroma_multiplier = (sat_bias - 1.0).max(0.0);
    let chromas: Vec<f32> = oklab_pts.iter().map(|p| p.chroma).collect();
    let mut weights: Vec<f32> = oklab_pixels
        .iter()
        .zip(chromas.iter())
        .map(|(p, &c)| {
            let mut w = 1.0 + (c * chroma_multiplier);
            if light_boost > 1.0 && p.l > 0.65 {
                let l_norm = (p.l - 0.65) / 0.35;
                w *= 1.0 + (light_boost - 1.0) * (l_norm * l_norm);
            }
            w
        })
        .collect();

    // 2. 3D (L, a, b) spatial histogram equalization.
    // Partitioning chromatic space into an 8x8x8 grid (512 bins) ensures that distinct
    // hue pockets (such as subtle blues in dark/gray scenes) are not smothered by massive
    // neutral gray clusters.
    let eq_power = (0.35_f32).max(0.85 - 0.4 / light_boost.max(0.1));

    let oklab_pixels_slice = &oklab_pixels[..n_unique];
    let weights_slice = &mut weights[..n_unique];
    let unique_counts_slice = &unique_counts[..n_unique];

    // --- Lightness histogram (16 bins) ---
    const L_BINS: usize = 16;
    let mut l_bin_weights = [0.0f32; L_BINS];
    for u in 0..n_unique {
        let bin = ((oklab_pixels_slice[u].l * L_BINS as f32) as usize).min(L_BINS - 1);
        let w = weights_slice[u] * (unique_counts_slice[u] as f32);
        l_bin_weights[bin] = l_bin_weights[bin].algebraic_add(w);
    }
    let active_l = l_bin_weights.iter().filter(|&&w| w > 0.0).count() as f32;
    let avg_l = if active_l > 0.0 { l_bin_weights.iter().sum::<f32>() / active_l } else { 1.0 };

    // --- 3D (L, a, b) Grid histogram (8x8x8 = 512 bins) ---
    let mut grid_weights = [0.0f32; 512];
    for u in 0..n_unique {
        let p = oklab_pixels_slice[u];
        let l_bin = ((p.l * 8.0) as usize).min(7);
        let a_bin = (((p.a + 0.3) * (8.0 / 0.6)) as usize).clamp(0, 7);
        let b_bin = (((p.b + 0.3) * (8.0 / 0.6)) as usize).clamp(0, 7);
        let bin = (l_bin << 6) | (a_bin << 3) | b_bin;
        let w = weights_slice[u] * (unique_counts_slice[u] as f32);
        grid_weights[bin] = grid_weights[bin].algebraic_add(w);
    }
    let active_grid = grid_weights.iter().filter(|&&w| w > 0.0).count() as f32;
    let avg_grid =
        if active_grid > 0.0 { grid_weights.iter().sum::<f32>() / active_grid } else { 1.0 };

    // --- Apply blended equalization ---
    // Keep 3D chromatic equalization active by default so subtle hues are preserved,
    // using lc_priority to smoothly shift priority from lightness-balance to pure hue-balance.
    let l_eff = 1.0 - (lc_priority * 0.5);
    let c_eff = 0.5 + (lc_priority * 0.5);

    for u in 0..n_unique {
        let p = oklab_pixels_slice[u];
        let l_bin = ((p.l * L_BINS as f32) as usize).min(L_BINS - 1);
        let gl_bin = ((p.l * 8.0) as usize).min(7);
        let a_bin = (((p.a + 0.3) * (8.0 / 0.6)) as usize).clamp(0, 7);
        let b_bin = (((p.b + 0.3) * (8.0 / 0.6)) as usize).clamp(0, 7);
        let g_bin = (gl_bin << 6) | (a_bin << 3) | b_bin;

        let l_eq = if l_bin_weights[l_bin] > 0.0 {
            (avg_l / l_bin_weights[l_bin]).powf(eq_power)
        } else {
            1.0
        };
        let c_eq = if grid_weights[g_bin] > 0.0 {
            (avg_grid / grid_weights[g_bin]).powf(eq_power)
        } else {
            1.0
        };

        weights_slice[u] *= l_eq.powf(l_eff) * c_eq.powf(c_eff);
    }

    // Precompute the effective weights array once to avoid millions of redundant multiplications
    let effective_weights: Vec<f32> =
        weights_slice.iter().zip(unique_counts_slice.iter()).map(|(&w, &c)| w * c as f32).collect();

    // 3. K-Means++ initialization using CentroidsSoA with multi-candidate local trial evaluation
    let mut centroids_soa = CentroidsSoA::with_capacity(k);

    // Pick first centroid: prioritize dynamic range highlights if light_boost > 1.2,
    // otherwise sample from the weighted distribution.
    let rng = SipRng::new(k, n_unique, rgb_pixels.len());
    let first_idx = if light_boost > 1.2 {
        oklab_pts
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.l.total_cmp(&b.1.l))
            .map(|(i, _)| i)
            .unwrap_or(0)
    } else {
        let total_w: f32 =
            effective_weights.iter().copied().fold(0.0f32, |acc, x| acc.algebraic_add(x));
        let mut threshold = rng.unit_f32(TAG_FIRST, 0) * total_w;
        let mut chosen_u = 0;
        for u in 0..n_unique {
            let w = effective_weights[u];
            threshold = threshold.algebraic_add(-w);
            if threshold <= 0.0 {
                chosen_u = u;
                break;
            }
        }
        chosen_u
    };
    centroids_soa.push(oklab_pts[first_idx]);

    let mut min_dists = vec![f32::MAX; n_unique];

    min_dists.par_iter_mut().enumerate().for_each(|(i, d)| {
        *d = oklch_weighted_dist_soa_single(oklab_pts[i], &centroids_soa, 0);
    });

    const CANDIDATES_PER_STEP: usize = 3;
    for ki in 1..k {
        let total: f32 = (0..n_unique)
            .into_par_iter()
            .map(|u| min_dists[u] * effective_weights[u])
            .reduce(|| 0.0f32, |a, b| a.algebraic_add(b));

        if total <= 0.0 {
            let idx = rng.below(TAG_FALLBACK, ki as u64, n_unique);
            centroids_soa.push(oklab_pts[idx]);
        } else {
            // Multi-trial candidate selection (k-means++ local trial heuristic)
            let mut best_candidate_idx = n_unique - 1;
            let mut max_cand_dist = -1.0f32;

            for trial in 0..CANDIDATES_PER_STEP {
                let sub_tag = TAG_TARGET_BASE + (trial as u8);
                let mut target = rng.unit_f32(sub_tag, ki as u64) * total;
                let mut cand_u = n_unique - 1;
                for u in 0..n_unique {
                    let w = min_dists[u] * effective_weights[u];
                    target = target.algebraic_add(-w);
                    if target <= 0.0 {
                        cand_u = u;
                        break;
                    }
                }
                if min_dists[cand_u] > max_cand_dist {
                    max_cand_dist = min_dists[cand_u];
                    best_candidate_idx = cand_u;
                }
            }
            centroids_soa.push(oklab_pts[best_candidate_idx]);
        }

        let new_c_idx = ki;
        min_dists.par_iter_mut().enumerate().for_each(|(i, d)| {
            let dist = oklch_weighted_dist_soa_single(oklab_pts[i], &centroids_soa, new_c_idx);
            if dist < *d {
                *d = dist;
            }
        });
    }

    // 4. K-Means iterations with persistent allocations
    let mut best_per_unique = vec![0u32; n_unique];

    // Chunking limits parallel split overhead & gives explicit cache locality
    let num_threads = rayon::current_num_threads().max(1);
    let chunk_size = n_unique.div_ceil(num_threads * 4).max(1);

    for _ in 0..20 {
        // Step A: Parallel nearest-centroid assignment using SoA
        best_per_unique.par_iter_mut().zip(min_dists.par_iter_mut()).enumerate().for_each(
            |(u, (best_idx, min_d))| {
                let p = oklab_pts[u];
                let mut min_dist = f32::MAX;
                let mut best = 0;
                // Unrolled SoA loop for contiguous vector loads
                for i in 0..centroids_soa.len() {
                    let dl = p.l - centroids_soa.l[i];
                    let dc = p.chroma - centroids_soa.chroma[i];
                    let da = p.a - centroids_soa.a[i];
                    let db = p.b - centroids_soa.b[i];
                    let dh_sq = (da * da + db * db - dc * dc).max(0.0);
                    let d = (dl * dl * 1.0) + (dc * dc * 2.0) + (dh_sq * 3.5);
                    if d < min_dist {
                        min_dist = d;
                        best = i;
                    }
                }
                *best_idx = best as u32;
                // Maintain `min_dists` accurately for zero-cost empty cluster reseeding
                *min_d = min_dist;
            },
        );

        // Step B: Parallel O(U) accumulation avoiding Rayon fold allocations
        let partials: Vec<(Vec<(f32, f32, f32)>, Vec<f32>)> = oklab_pixels_slice
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_idx, chunk)| {
                let mut sums = vec![(0.0f32, 0.0f32, 0.0f32); k];
                let mut counts = vec![0.0f32; k];
                for (u_in_chunk, p) in chunk.iter().enumerate() {
                    let u = chunk_idx * chunk_size + u_in_chunk;
                    let effective_w = effective_weights[u];
                    let best = best_per_unique[u] as usize;

                    sums[best].0 = sums[best].0.algebraic_add(p.l * effective_w);
                    sums[best].1 = sums[best].1.algebraic_add(p.a * effective_w);
                    sums[best].2 = sums[best].2.algebraic_add(p.b * effective_w);
                    counts[best] = counts[best].algebraic_add(effective_w);
                }
                (sums, counts)
            })
            .collect();

        let mut sums = vec![(0.0f32, 0.0f32, 0.0f32); k];
        let mut counts = vec![0.0f32; k];
        for (ps, pc) in &partials {
            for i in 0..k {
                sums[i].0 = sums[i].0.algebraic_add(ps[i].0);
                sums[i].1 = sums[i].1.algebraic_add(ps[i].1);
                sums[i].2 = sums[i].2.algebraic_add(ps[i].2);
                counts[i] = counts[i].algebraic_add(pc[i]);
            }
        }

        let mut max_shift = 0.0f32;
        for i in 0..k {
            if counts[i] > 0.0 {
                let new_ok = Oklab {
                    l: sums[i].0 / counts[i],
                    a: sums[i].1 / counts[i],
                    b: sums[i].2 / counts[i],
                };
                let new_ok_pt = OkPt::from_oklab(new_ok);
                let cur_pt = centroids_soa.get_pt(i);
                let shift = oklch_weighted_dist_pt(cur_pt, new_ok_pt);
                max_shift = max_shift.max(shift);
                centroids_soa.update(i, new_ok_pt);
            }
        }

        // Empty-cluster reseed intelligently using persistent `min_dists`
        // O(U) loop instead of O(U * K) network gathered per reseed.
        let mut reseeded = false;
        for empty_idx in 0..k {
            if counts[empty_idx] > 0.0 {
                continue;
            }

            let best_u = min_dists
                .par_iter()
                .enumerate()
                .map(|(i, &d)| (i, d))
                .max_by(|&(_, a), &(_, b)| a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);

            centroids_soa.update(empty_idx, oklab_pts[best_u]);
            // Avoid picking the identical distance pixel next loop
            min_dists[best_u] = 0.0;
            counts[empty_idx] = 1.0;
            reseeded = true;
        }

        if reseeded {
            max_shift = max_shift.max(1.0);
        }

        if max_shift < 1e-4 {
            break;
        }
    }

    // 5. Brightness preservation — correct systematic L drift before sRGB conversion
    let mut final_oklab = centroids_soa.to_oklab_vec();
    preserve_brightness(&mut final_oklab, rgb_pixels, brightness_preserve);

    // 6. Final mapping back to sRGB [u8; 3] with gamut preservation
    final_oklab.into_iter().map(oklab_to_srgb_u8).collect()
}

// -----------------------------------------------------------------------------
// Oklab Color Space Math Dependencies
// -----------------------------------------------------------------------------
#[derive(Clone, Copy, Debug)]
struct Oklab {
    l: f32,
    a: f32,
    b: f32,
}

/// Oklab point with its OkLCh chroma precomputed.
///
/// The weighted distance needs `sqrt(a² + b²)` for BOTH operands — but in
/// every hot loop one or both sides are loop-invariant (palette entries across
/// a scan, the pixel across its k comparisons). Precomputing chroma once per
/// point removes all square roots from the distance function. Bit-identical
/// to computing it inline per call: same op, same operands.
#[derive(Clone, Copy, Debug)]
struct OkPt {
    l: f32,
    a: f32,
    b: f32,
    chroma: f32, // == (a*a + b*b).sqrt()
}

impl OkPt {
    #[inline(always)]
    fn from_oklab(o: Oklab) -> Self {
        Self::from_arr([o.l, o.a, o.b])
    }

    #[inline(always)]
    fn from_arr(p: [f32; 3]) -> Self {
        Self { l: p[0], a: p[1], b: p[2], chroma: (p[1] * p[1] + p[2] * p[2]).sqrt() }
    }
}

/// The weighted OkLCh metric with both chromas precomputed — zero sqrts, pure
/// FMA. Same expression order as the original inline-sqrt version, so results
/// are bit-identical.
#[inline(always)]
fn oklch_weighted_dist_pt(p: OkPt, c: OkPt) -> f32 {
    // 1. Lightness difference
    let dl = p.l - c.l;
    // 2. Chroma difference
    let dc = p.chroma - c.chroma;
    // 3. Hue difference (chord length trick to avoid expensive atan2)
    let da = p.a - c.a;
    let db = p.b - c.b;
    let dh_sq = (da * da + db * db - dc * dc).max(0.0);

    // Perceptual Weights: 1.0 Lightness, 2.0 Chroma, 3.5 Hue — forces the
    // algorithm to preserve vibrant gradients and distinct minority hues (like dark blues)
    // instead of settling for mathematically safe but pale averages.
    (dl * dl * 1.0) + (dc * dc * 2.0) + (dh_sq * 3.5)
}

/// Squared Euclidean distance in u8-scale RGB — the metric for classic mode.
#[inline(always)]
fn rgb_dist_sq(p: [f32; 3], c: [f32; 3]) -> f32 {
    let dr = p[0] - c[0];
    let dg = p[1] - c[1];
    let db = p[2] - c[2];
    dr * dr + dg * dg + db * db
}

#[inline(always)]
fn rgb_to_f32(p: [u8; 3]) -> [f32; 3] {
    [p[0] as f32, p[1] as f32, p[2] as f32]
}

#[inline(always)]
fn srgb_to_oklab_arr(p: [u8; 3]) -> [f32; 3] {
    let o = srgb_to_oklab(p);
    [o.l, o.a, o.b]
}

/// Single shared Oklab → sRGB u8 conversion with gamut mapping.
/// If (L, a, b) lies outside the sRGB cube, reduces chroma towards neutral
/// at constant Lightness L to avoid blowing out or tinting bright details.
fn oklab_to_srgb_u8(mut ok: Oklab) -> [u8; 3] {
    let mut linear = oklab_to_linear_srgb(ok);
    if linear[0] >= 0.0
        && linear[0] <= 1.0
        && linear[1] >= 0.0
        && linear[1] <= 1.0
        && linear[2] >= 0.0
        && linear[2] <= 1.0
    {
        return [
            (linear_to_srgb(linear[0]) * 255.0).round() as u8,
            (linear_to_srgb(linear[1]) * 255.0).round() as u8,
            (linear_to_srgb(linear[2]) * 255.0).round() as u8,
        ];
    }

    // Out-of-gamut bisection: scale chroma (a, b) towards 0 preserving exact L
    let mut low = 0.0f32;
    let mut high = 1.0f32;
    let orig_a = ok.a;
    let orig_b = ok.b;

    for _ in 0..5 {
        let mid = (low + high) * 0.5;
        ok.a = orig_a * mid;
        ok.b = orig_b * mid;
        linear = oklab_to_linear_srgb(ok);
        if linear[0] >= 0.0
            && linear[0] <= 1.0
            && linear[1] >= 0.0
            && linear[1] <= 1.0
            && linear[2] >= 0.0
            && linear[2] <= 1.0
        {
            low = mid;
        } else {
            high = mid;
        }
    }
    ok.a = orig_a * low;
    ok.b = orig_b * low;
    linear = oklab_to_linear_srgb(ok);

    std::array::from_fn(|i| (linear_to_srgb(linear[i]).clamp(0.0, 1.0) * 255.0).round() as u8)
}

/// Precomputed lookup table: sRGB u8 → linear f32.
static SRGB_TO_LINEAR_LUT: std::sync::LazyLock<[f32; 256]> = std::sync::LazyLock::new(|| {
    let mut lut = [0.0f32; 256];
    for (i, v) in lut.iter_mut().enumerate() {
        let x = i as f32 / 255.0;
        *v = if x <= 0.04045 { x / 12.92 } else { ((x + 0.055) / 1.055).powf(2.4) };
    }
    lut
});

#[inline(always)]
fn srgb_u8_to_linear(v: u8) -> f32 {
    SRGB_TO_LINEAR_LUT[v as usize]
}

#[inline(always)]
fn linear_to_srgb(x: f32) -> f32 {
    if x <= 0.0031308 { x * 12.92 } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 }
}

fn srgb_to_oklab(rgb: [u8; 3]) -> Oklab {
    let r = srgb_u8_to_linear(rgb[0]);
    let g = srgb_u8_to_linear(rgb[1]);
    let b = srgb_u8_to_linear(rgb[2]);

    let l = 0.412_221_46 * r + 0.536_332_55 * g + 0.051_445_995 * b;
    let m = 0.211_903_5 * r + 0.680_699_5 * g + 0.107_396_96 * b;
    let s = 0.088_302_46 * r + 0.281_718_85 * g + 0.629_978_7 * b;

    let l_ = l.cbrt();
    let m_ = m.cbrt();
    let s_ = s.cbrt();

    Oklab {
        l: 0.210_454_26 * l_ + 0.793_617_8 * m_ - 0.004_072_047 * s_,
        a: 1.977_998_5 * l_ - 2.428_592_2 * m_ + 0.450_593_7 * s_,
        b: 0.025_904_037 * l_ + 0.782_771_77 * m_ - 0.808_675_77 * s_,
    }
}

fn oklab_to_linear_srgb(ok: Oklab) -> [f32; 3] {
    let l_ = ok.l + 0.396_337_78 * ok.a + 0.215_803_76 * ok.b;
    let m_ = ok.l - 0.105_561_346 * ok.a - 0.063_854_17 * ok.b;
    let s_ = ok.l - 0.089_484_18 * ok.a - 1.291_485_5 * ok.b;

    let l = l_.powi(3);
    let m = m_.powi(3);
    let s = s_.powi(3);

    [
        4.076_741_7 * l - 3.307_711_6 * m + 0.230_969_94 * s,
        -1.268_438 * l + 2.609_757_4 * m - 0.341_319_38 * s,
        -0.0041960863 * l - 0.703_418_6 * m + 1.707_614_7 * s,
    ]
}

// -----------------------------------------------------------------------------
// Blue Noise dependencies.
// Uses the same `blue-noise-256.bin` setup as `monochrome.rs`.
// -----------------------------------------------------------------------------
const NOISE_DATA: &[u8] = include_bytes!("blue-noise-256.bin");
const NOISE_DATA_WIDTH_AND_HEIGHT: usize = 256;
const _: () =
    assert!(NOISE_DATA.len() == NOISE_DATA_WIDTH_AND_HEIGHT * NOISE_DATA_WIDTH_AND_HEIGHT);

#[inline]
fn get_noise(x: u32, y: u32) -> u8 {
    let wrap_x = (x as usize) % NOISE_DATA_WIDTH_AND_HEIGHT;
    let wrap_y = (y as usize) % NOISE_DATA_WIDTH_AND_HEIGHT;
    NOISE_DATA[wrap_y * NOISE_DATA_WIDTH_AND_HEIGHT + wrap_x]
}
