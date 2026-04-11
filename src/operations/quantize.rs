use crate::{arg_parse_err::ArgParseErr, error::MagickError, image::Image, wm_err};
use image::{DynamicImage, RgbImage};
use rayon::prelude::*;

#[derive(Debug, Clone, PartialEq)]
pub struct QuantizeConfig {
    pub colors: u32,
    pub dither_level: f32, // 0.0 means disabled, 1.0 means full dither
    pub bias: f32, // <0.0 uses Oklab median-cut, 0.0 means classic k-means, >0.0 uses Oklab k-means++
    pub light_boost: f32, // >1.0 preserves bright detail, 1.0 is neutral (only used with Oklab k-means++)
    pub lc_priority: f32, // 0.0 = equalize lightness only, 1.0 = equalize chroma only, 0.5 = both equally
}

impl Default for QuantizeConfig {
    fn default() -> Self {
        Self {
            colors: 16,
            dither_level: 0.0,
            bias: 0.0,
            light_boost: 1.0,
            lc_priority: 0.0,
        }
    }
}

impl QuantizeConfig {
    /// Parse from a comma-separated string of three values, or "default".
    /// Format: "colors,dither_level,bias" or "colors,dither_level,bias:light_boost"
    /// or "colors,dither_level,bias:light_boost:lc_priority"
    ///
    ///   colors:  palette size (2-)
    ///   dither:  error diffusion strength (0.0 = off, 1.0 = full)
    ///   bias:    algorithm selector:
    ///         < 0.0  Oklab median-cut (fast, good for flat art)
    ///           0.0  RGB k-means
    ///         > 0.0  Oklab k-means++ (perceptual, value = saturation boost)
    ///
    ///  Oklab k-means++ accepts optional suffixes after bias:
    ///    bias:light_boost     highlight preservation (default 1.0, higher keeps brights)
    ///    bias:light_boost:lc  equalization blend (0.0 = lightness, 1.0 = chroma, default 0.0)
    ///
    ///  Examples:  16,1.0,0.0      16 colors, full dither, RGB k-means
    ///             8,0.5,-1.0       8 colors, half dither, Oklab median-cut
    ///             32,0.2,0.5:3.0   32 colors, Oklab k-means++, sat boost 0.5, light_boost 3.0
    ///           4,0.1,1.0:1.5:0.5  chroma+lightness equalization balanced
    pub fn parse_arg(s: &str) -> Result<Self, ArgParseErr> {
        let s = s.trim();
        if s.eq_ignore_ascii_case("default") {
            return Ok(Self::default());
        }

        let parts: Vec<&str> = s.split(',').collect();
        if parts.len() != 3 {
            return Err(ArgParseErr::with_msg(
                "quantize requires 'default' or exactly 3 comma-separated values: \
                 colors,dither_level,bias (bias may include :light_boost:lc_priority, e.g. 1.0:5.5:0.5)",
            ));
        }

        let colors = parts[0].trim().parse::<u32>().map_err(|_| {
            ArgParseErr::with_msg("invalid colors value (must be positive integer)")
        })?;

        let dither_level = parts[1]
            .trim()
            .parse::<f32>()
            .map_err(|_| ArgParseErr::with_msg("invalid dither_level value (must be float)"))?;

        // bias field supports optional :light_boost and :lc_priority suffixes
        // (e.g. "1.0:5.5" or "1.0:5.5:0.5")
        let bias_str = parts[2].trim();
        let colon_parts: Vec<&str> = bias_str.split(':').collect();
        let bias = colon_parts[0]
            .parse::<f32>()
            .map_err(|_| ArgParseErr::with_msg("invalid bias value (must be float)"))?;
        let light_boost = if colon_parts.len() > 1 {
            colon_parts[1]
                .parse::<f32>()
                .map_err(|_| ArgParseErr::with_msg("invalid light_boost value (must be float)"))?
        } else {
            1.0
        };
        let lc_priority = if colon_parts.len() > 2 {
            colon_parts[2]
                .parse::<f32>()
                .map_err(|_| {
                    ArgParseErr::with_msg("invalid lc_priority value (must be float 0.0-1.0)")
                })?
                .clamp(0.0, 1.0)
        } else {
            0.0
        };

        Ok(Self {
            colors,
            dither_level,
            bias,
            light_boost,
            lc_priority,
        })
    }
}

pub fn quantize(image: &mut Image, config: &QuantizeConfig) -> Result<(), MagickError> {
    if config.colors < 2 {
        return Err(wm_err!("quantize requires at least 2 colors"));
    }

    let input = image.pixels.to_rgb8();
    let width = input.width() as usize;

    // Flatten pixels to a slice of [u8; 3] for fast Rayon iteration
    let pixels: Vec<[u8; 3]> = input.pixels().map(|p| p.0).collect();

    // 1 & 2. Generate Palette using Classic RGB K-Means, Oklab K-Means++, or Oklab Median-Cut
    let palette = if config.bias < 0.0 {
        generate_palette_median_cut(&pixels, config.colors as usize)
    } else if config.bias > 0.0 {
        generate_palette_oklab(
            &pixels,
            config.colors as usize,
            config.bias,
            config.light_boost,
            config.lc_priority,
        )
    } else {
        generate_palette_rgb(&pixels, config.colors as usize)
    };

    // If something went horribly wrong and we have no palette, fallback safely
    if palette.is_empty() {
        return Err(wm_err!("quantize failed to generate a color palette"));
    }

    // 3. Map pixels to nearest palette color.
    // Two paths: error diffusion (when dithering is enabled) produces dramatically
    // better results for small palettes by propagating quantization error to
    // neighboring pixels. The parallel path is used when dithering is disabled.
    let mut output = RgbImage::new(input.width(), input.height());

    if config.dither_level > 0.0 {
        // ── Hybrid blue-noise + Sierra Lite error diffusion ──
        // Matches the approach in monochrome.rs: error diffusion preserves tonal
        // accuracy while blue noise breaks up the regular patterns that pure
        // error diffusion creates on smooth gradients.
        // Error buffer has +2 width padding for boundary-safe neighbor access.
        let padded_w = width + 2;
        let mut errors_r = vec![0.0f32; padded_w * (input.height() as usize)];
        let mut errors_g = vec![0.0f32; padded_w * (input.height() as usize)];
        let mut errors_b = vec![0.0f32; padded_w * (input.height() as usize)];

        // Blue noise amplitude: dither_level scales how much the noise
        // jitters the pixel before nearest-color search.
        let noise_spread = config.dither_level * 64.0;

        for y in 0..input.height() {
            for x in 0..input.width() {
                let i = (y as usize) * width + (x as usize);
                let idx = (y as usize) * padded_w + (x as usize) + 1;

                // Current pixel + accumulated diffused error, clamped to valid
                // range to prevent unbounded error accumulation that causes
                // bright areas to bleed large one-color blobs into dark areas.
                let r = (pixels[i][0] as f32 + errors_r[idx]).clamp(0.0, 255.0);
                let g = (pixels[i][1] as f32 + errors_g[idx]).clamp(0.0, 255.0);
                let b = (pixels[i][2] as f32 + errors_b[idx]).clamp(0.0, 255.0);

                // Blue noise jitter — shifts decision boundaries between palette
                // colors, breaking up the structured patterns of pure error diffusion.
                let noise_u8 = get_noise(x, y);
                let noise = (noise_u8 as f32 / 255.0) - 0.5;
                let jitter = noise * noise_spread;

                let jr = (r + jitter).clamp(0.0, 255.0);
                let jg = (g + jitter).clamp(0.0, 255.0);
                let jb = (b + jitter).clamp(0.0, 255.0);

                // Find nearest palette color using the jittered values
                let mut min_dist = f32::MAX;
                let mut best_color = palette[0];
                for color in &palette {
                    let dr = jr - color[0] as f32;
                    let dg = jg - color[1] as f32;
                    let db = jb - color[2] as f32;
                    let dist = dr * dr + dg * dg + db * db;
                    if dist < min_dist {
                        min_dist = dist;
                        best_color = *color;
                    }
                }

                output.put_pixel(x, y, image::Rgb(best_color));

                // Quantization error computed from the jittered values — the same
                // values used for palette selection — so there is no systematic
                // bias between what was chosen and what error is propagated.
                let err_r = jr - best_color[0] as f32;
                let err_g = jg - best_color[1] as f32;
                let err_b = jb - best_color[2] as f32;

                // Sierra Lite distribution (same as monochrome.rs):
                //   current → [right: 2/4, bottom-left: 1/4, bottom: 1/4]
                if x < input.width() - 1 {
                    errors_r[idx + 1] += err_r * 0.5;
                    errors_g[idx + 1] += err_g * 0.5;
                    errors_b[idx + 1] += err_b * 0.5;
                }
                if y < input.height() - 1 {
                    let below = idx + padded_w;
                    errors_r[below - 1] += err_r * 0.25;
                    errors_g[below - 1] += err_g * 0.25;
                    errors_b[below - 1] += err_b * 0.25;
                    errors_r[below] += err_r * 0.25;
                    errors_g[below] += err_g * 0.25;
                    errors_b[below] += err_b * 0.25;
                }
            }
        }
    } else {
        // ── No dithering: parallel nearest-color mapping ──
        let raw_out = output.as_mut();
        raw_out
            .par_chunks_exact_mut(3)
            .enumerate()
            .for_each(|(i, pixel_out)| {
                let r = pixels[i][0] as i32;
                let g = pixels[i][1] as i32;
                let b = pixels[i][2] as i32;

                let mut min_dist = u32::MAX;
                let mut best_color = palette[0];

                for color in &palette {
                    let dr = r - color[0] as i32;
                    let dg = g - color[1] as i32;
                    let db = b - color[2] as i32;
                    let dist = (dr * dr + dg * dg + db * db) as u32;

                    if dist < min_dist {
                        min_dist = dist;
                        best_color = *color;
                    }
                }

                pixel_out[0] = best_color[0];
                pixel_out[1] = best_color[1];
                pixel_out[2] = best_color[2];
            });
    }

    image.pixels = DynamicImage::ImageRgb8(output);
    Ok(())
}

/// Original Fast RGB K-Means (used when bias == 0.0)
fn generate_palette_rgb(pixels: &[[u8; 3]], k: usize) -> Vec<[u8; 3]> {
    let mut centroids = vec![[0.0f32; 3]; k];
    let step = (pixels.len() / k).max(1);
    for i in 0..k {
        let sample_idx = (i * step).min(pixels.len() - 1);
        let p = pixels[sample_idx];
        centroids[i] = [p[0] as f32, p[1] as f32, p[2] as f32];
    }

    let max_iterations = 15;
    for _ in 0..max_iterations {
        let (new_sums, new_counts) = pixels
            .par_iter()
            .fold(
                || (vec![[0.0f32; 3]; k], vec![0usize; k]),
                |mut acc, p| {
                    let mut min_dist = f32::MAX;
                    let mut best_idx = 0;
                    for (i, c) in centroids.iter().enumerate() {
                        let dr = p[0] as f32 - c[0];
                        let dg = p[1] as f32 - c[1];
                        let db = p[2] as f32 - c[2];
                        let dist = (dr * dr) + (dg * dg) + (db * db);
                        if dist < min_dist {
                            min_dist = dist;
                            best_idx = i;
                        }
                    }
                    acc.0[best_idx][0] += p[0] as f32;
                    acc.0[best_idx][1] += p[1] as f32;
                    acc.0[best_idx][2] += p[2] as f32;
                    acc.1[best_idx] += 1;
                    acc
                },
            )
            .reduce(
                || (vec![[0.0f32; 3]; k], vec![0usize; k]),
                |mut a, b| {
                    for i in 0..k {
                        a.0[i][0] += b.0[i][0];
                        a.0[i][1] += b.0[i][1];
                        a.0[i][2] += b.0[i][2];
                        a.1[i] += b.1[i];
                    }
                    a
                },
            );

        let mut changed = false;
        for i in 0..k {
            if new_counts[i] > 0 {
                let count = new_counts[i] as f32;
                let new_r = new_sums[i][0] / count;
                let new_g = new_sums[i][1] / count;
                let new_b = new_sums[i][2] / count;

                if (centroids[i][0] - new_r).abs() > 0.5
                    || (centroids[i][1] - new_g).abs() > 0.5
                    || (centroids[i][2] - new_b).abs() > 0.5
                {
                    changed = true;
                }

                centroids[i] = [new_r, new_g, new_b];
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
                c[0].clamp(0.0, 255.0) as u8,
                c[1].clamp(0.0, 255.0) as u8,
                c[2].clamp(0.0, 255.0) as u8,
            ]
        })
        .collect()
}

/// Median-cut palette generation in Oklab perceptual space (used when bias < 0.0).
/// Recursively splits the color space by the channel with highest variance,
/// which naturally preserves dark tones and minority color clusters.
fn generate_palette_median_cut(rgb_pixels: &[[u8; 3]], k: usize) -> Vec<[u8; 3]> {
    let k = k.max(1);
    let mut oklab_pixels: Vec<Oklab> = rgb_pixels.par_iter().map(|&p| srgb_to_oklab(p)).collect();

    if oklab_pixels.is_empty() {
        return vec![[0, 0, 0]; k];
    }

    // Each box is a contiguous range in the pixel array. After sorting a box
    // by its highest-variance channel, splitting at the median yields two child
    // boxes that are themselves contiguous — no copies needed.
    struct McBox {
        start: usize,
        len: usize,
        /// Which Oklab channel (0=L, 1=a, 2=b) has the highest weighted variance
        split_channel: usize,
        /// Total weighted variance across all channels — used as the priority key.
        /// Multiplied by pixel count so large, varied boxes are split first.
        priority: f32,
    }

    // Threshold: use par_sort for boxes larger than this
    const PAR_SORT_THRESHOLD: usize = 50_000;

    // Oklab's a/b channels have a much narrower natural range (~±0.3) than L (0–1).
    // Without scaling, L dominates all splits and chromatic differences (skin hues,
    // blue vs. gray) are ignored. These weights equalize the channels' contribution
    // to variance so that hue/chroma splits happen when they should.
    const CHANNEL_WEIGHT: [f32; 3] = [1.0, 3.0, 3.0];

    let compute_box_stats = |pixels: &[Oklab], start: usize, len: usize| -> McBox {
        let slice = &pixels[start..start + len];
        let n = len as f32;

        // Compute mean for each channel
        let mut sum = [0.0f32; 3];
        for p in slice {
            sum[0] += p.l;
            sum[1] += p.a;
            sum[2] += p.b;
        }
        let mean = [sum[0] / n, sum[1] / n, sum[2] / n];

        // Compute variance for each channel
        let mut var = [0.0f32; 3];
        for p in slice {
            let vals = [p.l, p.a, p.b];
            for ch in 0..3 {
                let d = vals[ch] - mean[ch];
                var[ch] += d * d;
            }
        }
        // Weighted variance (not divided by n — we want total, not average,
        // so large boxes naturally get higher priority)
        let weighted_var = [
            var[0] * CHANNEL_WEIGHT[0],
            var[1] * CHANNEL_WEIGHT[1],
            var[2] * CHANNEL_WEIGHT[2],
        ];

        let (split_channel, _) = weighted_var
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        let total_priority = weighted_var[0] + weighted_var[1] + weighted_var[2];

        McBox {
            start,
            len,
            split_channel,
            priority: total_priority,
        }
    };

    // Start with one box containing all pixels
    let initial = compute_box_stats(&oklab_pixels, 0, oklab_pixels.len());
    let mut boxes: Vec<McBox> = vec![initial];

    while boxes.len() < k {
        // Find the box with the highest total weighted variance to split
        let best_idx = boxes
            .iter()
            .enumerate()
            .filter(|(_, b)| b.len >= 2)
            .max_by(|a, b| {
                a.1.priority
                    .partial_cmp(&b.1.priority)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i);

        let best_idx = match best_idx {
            Some(i) => i,
            None => break, // all boxes have 1 pixel, can't split further
        };

        let bx = boxes.swap_remove(best_idx);
        let ch = bx.split_channel;

        // Sort the box's slice by the chosen channel
        let slice = &mut oklab_pixels[bx.start..bx.start + bx.len];
        if slice.len() >= PAR_SORT_THRESHOLD {
            match ch {
                0 => slice.par_sort_unstable_by(|a, b| a.l.total_cmp(&b.l)),
                1 => slice.par_sort_unstable_by(|a, b| a.a.total_cmp(&b.a)),
                _ => slice.par_sort_unstable_by(|a, b| a.b.total_cmp(&b.b)),
            }
        } else {
            match ch {
                0 => slice.sort_unstable_by(|a, b| a.l.total_cmp(&b.l)),
                1 => slice.sort_unstable_by(|a, b| a.a.total_cmp(&b.a)),
                _ => slice.sort_unstable_by(|a, b| a.b.total_cmp(&b.b)),
            }
        }

        // Split at the median
        let mid = bx.len / 2;
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
            let (sum_l, sum_a, sum_b) = slice
                .iter()
                .fold((0.0f64, 0.0f64, 0.0f64), |(sl, sa, sb), p| {
                    (sl + p.l as f64, sa + p.a as f64, sb + p.b as f64)
                });
            let avg = Oklab {
                l: (sum_l / n) as f32,
                a: (sum_a / n) as f32,
                b: (sum_b / n) as f32,
            };
            let linear = oklab_to_linear_srgb(avg);
            [
                (linear_to_srgb(linear[0]).clamp(0.0, 1.0) * 255.0).round() as u8,
                (linear_to_srgb(linear[1]).clamp(0.0, 1.0) * 255.0).round() as u8,
                (linear_to_srgb(linear[2]).clamp(0.0, 1.0) * 255.0).round() as u8,
            ]
        })
        .collect()
}

/// K-means++ clustering in Oklab with Oklch distance (used when bias > 0.0).
///
/// Histogram equalization across lightness and chroma ensures dark areas and
/// minority colors get fair palette representation. sat_bias boosts saturated
/// pixels' weight, light_boost controls equalization strength, and lc_priority
/// blends between lightness-only (0.0) and chroma-only (1.0) equalization.
fn generate_palette_oklab(
    rgb_pixels: &[[u8; 3]],
    k: usize,
    sat_bias: f32,
    light_boost: f32,
    lc_priority: f32,
) -> Vec<[u8; 3]> {
    let k = k.max(1);
    let oklab_pixels: Vec<Oklab> = rgb_pixels.par_iter().map(|&p| srgb_to_oklab(p)).collect();

    if oklab_pixels.is_empty() {
        return vec![[0, 0, 0]; k];
    }

    // 1. Per-pixel weights: uniform base with chroma boost controlled by sat_bias.
    //    No pixels are culled — every pixel participates in clustering so the
    //    final palette can represent the full image faithfully.
    let chromas: Vec<f32> = oklab_pixels
        .iter()
        .map(|p| (p.a * p.a + p.b * p.b).sqrt())
        .collect();

    let mut weights: Vec<f32> = chromas
        .iter()
        .map(|&c| 1.0 + (c * 15.0).powf(1.5) * sat_bias)
        .collect();

    // 2. Histogram equalization across lightness and chroma.
    //    Two independent 16-bin histograms are computed, and their equalization
    //    factors are blended via lc_priority: 0.0 = lightness only, 1.0 = chroma
    //    only, 0.5 = both equally. This is what preserves dark-area detail —
    //    underrepresented lightness bins get boosted so k-means allocates
    //    palette slots to them proportionally.
    const LC_BINS: usize = 16;

    // Equalization strength scales with light_boost
    let eq_power = (0.5_f32).max(1.0 - 0.5 / light_boost.max(0.1));

    // --- Lightness histogram ---
    let mut l_bin_weights = [0.0f32; LC_BINS];
    for (p, &w) in oklab_pixels.iter().zip(weights.iter()) {
        let bin = ((p.l * LC_BINS as f32) as usize).min(LC_BINS - 1);
        l_bin_weights[bin] += w;
    }
    let active_l = l_bin_weights.iter().filter(|&&w| w > 0.0).count() as f32;
    let avg_l = if active_l > 0.0 {
        l_bin_weights.iter().sum::<f32>() / active_l
    } else {
        1.0
    };

    // --- Chroma histogram ---
    // Find max chroma to scale bins across the image's actual range
    let max_chroma = chromas
        .iter()
        .copied()
        .max_by(|a, b| a.total_cmp(b))
        .unwrap_or(0.0);
    let chroma_scale = if max_chroma > 1e-6 {
        LC_BINS as f32 / max_chroma
    } else {
        1.0
    };

    let mut c_bin_weights = [0.0f32; LC_BINS];
    for (&c, &w) in chromas.iter().zip(weights.iter()) {
        let bin = ((c * chroma_scale) as usize).min(LC_BINS - 1);
        c_bin_weights[bin] += w;
    }
    let active_c = c_bin_weights.iter().filter(|&&w| w > 0.0).count() as f32;
    let avg_c = if active_c > 0.0 {
        c_bin_weights.iter().sum::<f32>() / active_c
    } else {
        1.0
    };

    // --- Apply blended equalization ---
    let l_weight = 1.0 - lc_priority;
    let c_weight = lc_priority;
    for (i, w) in weights.iter_mut().enumerate() {
        let l_bin = ((oklab_pixels[i].l * LC_BINS as f32) as usize).min(LC_BINS - 1);
        let c_bin = ((chromas[i] * chroma_scale) as usize).min(LC_BINS - 1);

        let l_eq = if l_bin_weights[l_bin] > 0.0 {
            (avg_l / l_bin_weights[l_bin]).powf(eq_power)
        } else {
            1.0
        };
        let c_eq = if c_bin_weights[c_bin] > 0.0 {
            (avg_c / c_bin_weights[c_bin]).powf(eq_power)
        } else {
            1.0
        };

        // Geometric blend: eq = l_eq^(1-priority) * c_eq^priority
        *w *= l_eq.powf(l_weight) * c_eq.powf(c_weight);
    }

    // 3. Oklch distance metric — atan2-free using chord-based hue difference.
    //    Uses: ΔH² = Δa² + Δb² - ΔC² (standard CIEDE trick to avoid atan2).
    //    Then scales by effective_chroma² to match original weighting intent.
    #[inline(always)]
    fn oklch_dist(p: Oklab, p_c: f32, c: Oklab, c_c: f32) -> f32 {
        let mut effective_chroma = p_c.max(c_c);

        // Only apply the wedge to dark colors if they ACTUALLY have some color.
        // If it's practically pure grey (chroma < 0.015), let it stay grey!
        if p.l < 0.6 && c.l < 0.6 && effective_chroma > 0.015 {
            effective_chroma = effective_chroma.max(0.04);
        }
        effective_chroma = effective_chroma.min(0.25);

        // Lightness
        let dl = (p.l - c.l) * 2.0;
        // Chroma distance
        let delta_c = p_c - c_c;
        let dc = delta_c * 4.0;

        // Chord-based hue difference: ΔH² = Δa² + Δb² - ΔC²
        let da = p.a - c.a;
        let db = p.b - c.b;
        let dh_sq = (da * da + db * db - delta_c * delta_c).max(0.0);

        // For perceptual equivalence we scale chord² by (eff_chroma² * 9) / max(Cp*Cc, ε).
        let cc_product = (p_c * c_c).max(1e-8);
        let dh_weighted_sq = dh_sq * (effective_chroma * effective_chroma * 9.0) / cc_product;

        dl * dl + dc * dc + dh_weighted_sq
    }

    // 4. K-Means++ initialization
    let mut centroids = vec![oklab_pixels[0]; k];
    let mut centroid_chromas = vec![0.0f32; k];
    let mut min_dists = vec![f32::MAX; oklab_pixels.len()]; // Cache nearest centroid distance

    let mut rng_state: u64 = 0x5EED_C0DE_1234_5678;
    let mut xorshift = || -> u64 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        rng_state
    };

    let total_w: f32 = weights.iter().sum();
    let mut threshold = (xorshift() as f32 / u64::MAX as f32) * total_w;
    for (i, &w) in weights.iter().enumerate() {
        threshold -= w;
        if threshold <= 0.0 {
            centroids[0] = oklab_pixels[i];
            break;
        }
    }

    centroid_chromas[0] =
        (centroids[0].a * centroids[0].a + centroids[0].b * centroids[0].b).sqrt();

    // Initial pass: populate min_dists for the first centroid
    min_dists.par_iter_mut().enumerate().for_each(|(i, d)| {
        *d = oklch_dist(
            oklab_pixels[i],
            chromas[i],
            centroids[0],
            centroid_chromas[0],
        );
    });

    for ki in 1..k {
        // Calculate the weighted sum using cached min_dists.
        // Deterministic parallel sum: fixed-size chunks are summed in
        // parallel, then the chunk totals are combined sequentially in
        // index order. This avoids FP non-associativity jitter while
        // staying parallel for large pixel counts.
        let total: f32 = min_dists
            .par_chunks(4096)
            .zip(weights.par_chunks(4096))
            .map(|(d_chunk, w_chunk)| {
                d_chunk
                    .iter()
                    .zip(w_chunk.iter())
                    .map(|(&d, &w)| d * w)
                    .sum::<f32>()
            })
            .collect::<Vec<f32>>()
            .iter()
            .sum();

        if total <= 0.0 {
            centroids[ki] = oklab_pixels[(xorshift() as usize) % oklab_pixels.len()];
        } else {
            let mut target = (xorshift() as f32 / u64::MAX as f32) * total;
            for i in 0..oklab_pixels.len() {
                target -= min_dists[i] * weights[i];
                if target <= 0.0 {
                    centroids[ki] = oklab_pixels[i];
                    break;
                }
            }
        }

        // Precompute centroid chromas for this round
        centroid_chromas[ki] =
            (centroids[ki].a * centroids[ki].a + centroids[ki].b * centroids[ki].b).sqrt();
        let new_c = centroids[ki];
        let new_c_chroma = centroid_chromas[ki];

        // Update cached distances, ONLY checking against the newly added centroid
        min_dists.par_iter_mut().enumerate().for_each(|(i, d)| {
            let dist = oklch_dist(oklab_pixels[i], chromas[i], new_c, new_c_chroma);
            if dist < *d {
                *d = dist;
            }
        });
    }

    // 5. K-Means iterations
    for _ in 0..20 {
        // Pre-pack ONLY the centroids.
        #[derive(Clone, Copy)]
        struct CentroidData {
            l: f32,
            a: f32,
            b: f32,
            c: f32,
            inv_c: f32,
            is_dark: bool,
        }

        let cent_data: Vec<CentroidData> = centroids
            .iter()
            .map(|c| {
                let ch = (c.a * c.a + c.b * c.b).sqrt();
                CentroidData {
                    l: c.l,
                    a: c.a,
                    b: c.b,
                    c: ch,
                    inv_c: 1.0 / ch.max(1e-4), // Precalculate centroid division
                    is_dark: c.l < 0.6,
                }
            })
            .collect();

        // Extract slice so the compiler doesn't insert bounds checks in the inner loop
        let cd_slice = &cent_data[..];

        // Deterministic parallel accumulation: fixed-size chunks are processed
        // in parallel and collected in index order, so the final sequential
        // reduction always combines partial sums identically across runs.
        // This avoids FP non-associativity jitter from Rayon's work-stealing.
        const CHUNK: usize = 4096;
        let n_pixels = oklab_pixels.len();
        let n_chunks = (n_pixels + CHUNK - 1) / CHUNK;
        let partials: Vec<(Vec<(f32, f32, f32)>, Vec<f32>)> = (0..n_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let start = chunk_idx * CHUNK;
                let end = (start + CHUNK).min(n_pixels);
                let mut sums = vec![(0.0f32, 0.0f32, 0.0f32); k];
                let mut counts = vec![0.0f32; k];
                for i in start..end {
                    let p = &oklab_pixels[i];
                    let p_c = chromas[i];
                    let w = weights[i];

                    let mut min_dist = f32::MAX;
                    let mut best_idx = 0;

                    // Hoist pixel properties out of the centroid loop
                    let p_l = p.l;
                    let p_a = p.a;
                    let p_b = p.b;
                    let p_is_dark = p_l < 0.6;

                    // Precalculate pixel division OUTSIDE the inner loop
                    let inv_p_c = 1.0 / p_c.max(1e-4);

                    for (ci, cd) in cd_slice.iter().enumerate() {
                        // ── EARLY EXIT 1: Lightness ──
                        let dl = p_l - cd.l;
                        let dl_sq = dl * dl * 4.0;
                        if dl_sq >= min_dist {
                            continue;
                        } // Lightness difference alone is worse

                        // ── EARLY EXIT 2: Chroma ──
                        let delta_c = p_c - cd.c;
                        let dc_sq = delta_c * delta_c * 16.0;
                        let base_dist = dl_sq + dc_sq;
                        if base_dist >= min_dist {
                            continue;
                        } // Lightness + Chroma is worse

                        let mut eff_chroma = p_c.max(cd.c);
                        if p_is_dark && cd.is_dark && eff_chroma > 0.015 {
                            eff_chroma = eff_chroma.max(0.04);
                        }
                        eff_chroma = eff_chroma.min(0.25);

                        let da = p_a - cd.a;
                        let db = p_b - cd.b;
                        let dh_sq = da * da + db * db - delta_c * delta_c;
                        let dh_sq_pos = dh_sq.max(0.0);

                        let inv_cc = inv_p_c * cd.inv_c;
                        let dh_weighted_sq = dh_sq_pos * (eff_chroma * eff_chroma * 9.0) * inv_cc;

                        let dist_sq = base_dist + dh_weighted_sq;
                        if dist_sq < min_dist {
                            min_dist = dist_sq;
                            best_idx = ci;
                        }
                    }

                    sums[best_idx].0 += p_l * w;
                    sums[best_idx].1 += p_a * w;
                    sums[best_idx].2 += p_b * w;
                    counts[best_idx] += w;
                }
                (sums, counts)
            })
            .collect();

        // Sequential reduction in fixed order → deterministic
        let mut sums = vec![(0.0f32, 0.0f32, 0.0f32); k];
        let mut counts = vec![0.0f32; k];
        for (ps, pc) in &partials {
            for i in 0..k {
                sums[i].0 += ps[i].0;
                sums[i].1 += ps[i].1;
                sums[i].2 += ps[i].2;
                counts[i] += pc[i];
            }
        }

        let mut max_shift = 0.0f32;
        for i in 0..k {
            if counts[i] > 0.0 {
                let new_l = sums[i].0 / counts[i];
                let new_a = sums[i].1 / counts[i];
                let new_b = sums[i].2 / counts[i];

                let new_ok = Oklab {
                    l: new_l,
                    a: new_a,
                    b: new_b,
                };
                let old_c =
                    (centroids[i].a * centroids[i].a + centroids[i].b * centroids[i].b).sqrt();
                let new_c = (new_a * new_a + new_b * new_b).sqrt();
                let shift = oklch_dist(centroids[i], old_c, new_ok, new_c);

                max_shift = max_shift.max(shift);
                centroids[i] = new_ok;
            }
        }
        if max_shift < 1e-6 {
            break;
        }
    }

    // 6. Final mapping back to sRGB [u8; 3]
    centroids
        .into_iter()
        .map(|ok_c| {
            let linear = oklab_to_linear_srgb(ok_c);
            [
                (linear_to_srgb(linear[0]).clamp(0.0, 1.0) * 255.0).round() as u8,
                (linear_to_srgb(linear[1]).clamp(0.0, 1.0) * 255.0).round() as u8,
                (linear_to_srgb(linear[2]).clamp(0.0, 1.0) * 255.0).round() as u8,
            ]
        })
        .collect()
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

/// Precomputed lookup table: sRGB u8 → linear f32.
static SRGB_TO_LINEAR_LUT: std::sync::LazyLock<[f32; 256]> = std::sync::LazyLock::new(|| {
    let mut lut = [0.0f32; 256];
    for i in 0..256 {
        let x = i as f32 / 255.0;
        lut[i] = if x <= 0.04045 {
            x / 12.92
        } else {
            ((x + 0.055) / 1.055).powf(2.4)
        };
    }
    lut
});

#[inline(always)]
fn srgb_u8_to_linear(v: u8) -> f32 {
    SRGB_TO_LINEAR_LUT[v as usize]
}

#[inline(always)]
fn linear_to_srgb(x: f32) -> f32 {
    if x <= 0.0031308 {
        x * 12.92
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
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

#[inline]
fn get_noise(x: u32, y: u32) -> u8 {
    let wrap_x = (x as usize) % NOISE_DATA_WIDTH_AND_HEIGHT;
    let wrap_y = (y as usize) % NOISE_DATA_WIDTH_AND_HEIGHT;
    NOISE_DATA[wrap_y * NOISE_DATA_WIDTH_AND_HEIGHT + wrap_x]
}
