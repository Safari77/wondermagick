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
    let height = input.height() as usize;

    // Flatten pixels to a slice of [u8; 3] for fast Rayon iteration
    let pixels: Vec<[u8; 3]> = input.pixels().map(|p| p.0).collect();

    // 1 & 2. Generate Palette using MacQueen K-Means, Classic RGB K-Means, Oklab K-Means++, or Oklab Median-Cut
    let palette = if config.bias <= -2.0 {
        generate_palette_macqueen(&pixels, config.colors as usize, width, config.dither_level)
    } else if config.bias < 0.0 {
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

    // Palette cached in Oklab once. Mapping uses the same oklch_weighted_dist
    // the palette was generated under, so the selection metric is consistent end-to-end.
    let palette_oklab: Vec<Oklab> = palette.iter().map(|&c| srgb_to_oklab(c)).collect();

    if config.dither_level > 0.0 {
        // ── Hybrid blue-noise + Sierra Lite error diffusion ──
        // Matches the approach in monochrome.rs: error diffusion preserves tonal
        // accuracy while blue noise breaks up the regular patterns that pure
        // error diffusion creates on smooth gradients.
        // Diffusion runs in Oklab so the propagated "error" is perceptually
        // meaningful and lives in the same space as the selection metric.
        let padded_w = width + 2;
        let mut errors_l = vec![0.0f32; padded_w * height];
        let mut errors_a = vec![0.0f32; padded_w * height];
        let mut errors_b = vec![0.0f32; padded_w * height];

        // Convert source image to Oklab once. Per-pixel dither state is then
        // pixels_oklab[i] + accumulated diffused error + jitter.
        let pixels_oklab: Vec<Oklab> = pixels.par_iter().map(|&p| srgb_to_oklab(p)).collect();

        // Blue noise amplitude: dither_level scales how much the noise
        // jitters the pixel before nearest-color search.
        let noise_spread = config.dither_level * 0.1;

        // Per-pixel error clamp applied at READ.
        const ERR_CLAMP_L: f32 = 0.2;
        const ERR_CLAMP_AB: f32 = 0.05;

        for y in 0..height {
            // Serpentine: even rows scan left→right, odd rows right→left.
            let ltr = y % 2 == 0;
            let fwd_dx: i32 = if ltr { 1 } else { -1 };
            let diag_dx: i32 = -fwd_dx;

            for step in 0..width {
                let xs = if ltr { step } else { width - 1 - step };
                let i = y * width + xs;
                let idx = y * padded_w + xs + 1;

                // Current pixel + accumulated diffused error, clamped to valid
                // range to prevent unbounded error accumulation that causes
                // bright areas to bleed large one-color blobs into dark areas.
                let p = pixels_oklab[i];
                let el = errors_l[idx].clamp(-ERR_CLAMP_L, ERR_CLAMP_L);
                let ea = errors_a[idx].clamp(-ERR_CLAMP_AB, ERR_CLAMP_AB);
                let eb = errors_b[idx].clamp(-ERR_CLAMP_AB, ERR_CLAMP_AB);

                // Blue noise jitter — shifts decision boundaries between palette
                // colors, breaking up the structured patterns of pure error diffusion.
                // Applied along L (the gray-axis analogue in Oklab), matching the
                // effect of the previous sRGB same-value-on-R/G/B jitter.
                let noise_u8 = get_noise(xs as u32, y as u32);
                let noise = (noise_u8 as f32 / 255.0) - 0.5;
                let jitter = noise * noise_spread;

                let jp = Oklab {
                    l: p.l + el + jitter,
                    a: p.a + ea,
                    b: p.b + eb,
                };

                // Find nearest palette color using the jittered values
                let mut min_dist = f32::MAX;
                let mut best_idx: usize = 0;
                for (ci, &c_ok) in palette_oklab.iter().enumerate() {
                    let d = oklch_weighted_dist(jp, c_ok);
                    if d < min_dist {
                        min_dist = d;
                        best_idx = ci;
                    }
                }

                output.put_pixel(xs as u32, y as u32, image::Rgb(palette[best_idx]));

                // Quantization error EXCLUDES the jitter term. Jitter's role is to
                // perturb which palette entry wins (it's the whole point of blue
                // noise here), but propagating jitter into the error buffer turns
                // that buffer into a bounded random walk across flat regions — and
                // on long runs the walk eventually reaches the clamp limit, at
                // which point (source + clamped_err + jitter) lands close enough
                // to a distant palette entry (dark blue → skin-tone red) that the
                // wrong entry wins for a single pixel before the correction
                // propagates. Using (p + el) here matches the no-jitter path's
                // error formula exactly.
                let chosen = palette_oklab[best_idx];
                let err_l = (p.l + el) - chosen.l;
                let err_a = (p.a + ea) - chosen.a;
                let err_b = (p.b + eb) - chosen.b;

                // Sierra Lite distribution (same as monochrome.rs):
                //   current → [right: 2/4, bottom-left: 1/4, bottom: 1/4]
                // On R→L rows the pattern is mirrored:
                //   current → [left: 2/4, bottom-right: 1/4, bottom: 1/4]
                // so error always propagates in the scan direction. Edge
                // pixels write into the left/right padding column (which is
                // never read), so no explicit x-bounds check is needed.
                let fwd_col = (xs as i32 + fwd_dx + 1) as usize;
                let fwd_idx = y * padded_w + fwd_col;
                errors_l[fwd_idx] += err_l * 0.5;
                errors_a[fwd_idx] += err_a * 0.5;
                errors_b[fwd_idx] += err_b * 0.5;

                if y + 1 < height {
                    let below_row = (y + 1) * padded_w;
                    let diag_col = (xs as i32 + diag_dx + 1) as usize;
                    let diag_idx = below_row + diag_col;
                    errors_l[diag_idx] += err_l * 0.25;
                    errors_a[diag_idx] += err_a * 0.25;
                    errors_b[diag_idx] += err_b * 0.25;
                    let below_idx = below_row + xs + 1;
                    errors_l[below_idx] += err_l * 0.25;
                    errors_a[below_idx] += err_a * 0.25;
                    errors_b[below_idx] += err_b * 0.25;
                }
            }
        }
    } else {
        // ── No dithering: parallel nearest-color mapping ──
        // Each pixel is converted to Oklab and matched against palette_oklab
        // under the same weighted distance used for palette generation.
        let raw_out = output.as_mut();
        raw_out
            .par_chunks_exact_mut(3)
            .enumerate()
            .for_each(|(i, pixel_out)| {
                let p = srgb_to_oklab(pixels[i]);

                let mut min_dist = f32::MAX;
                let mut best_idx: usize = 0;

                for (ci, &c_ok) in palette_oklab.iter().enumerate() {
                    let d = oklch_weighted_dist(p, c_ok);
                    if d < min_dist {
                        min_dist = d;
                        best_idx = ci;
                    }
                }

                let best_color = palette[best_idx];
                pixel_out[0] = best_color[0];
                pixel_out[1] = best_color[1];
                pixel_out[2] = best_color[2];
            });
    }

    image.pixels = DynamicImage::ImageRgb8(output);
    Ok(())
}

/// MacQueen K-Means clustering natively in Oklab perceptual space.
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

    // 1. Thread-local compatible fast xorshift PRNG
    let mut rng_state: u64 = 0x1234_BEEF;
    let mut xorshift = || -> u64 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        rng_state
    };

    // 2. Initial centroids
    let mut centroids = vec![
        Oklab {
            l: 0.0,
            a: 0.0,
            b: 0.0
        };
        k
    ];
    for c in &mut centroids {
        let idx = (xorshift() as usize) % n_pixels;
        *c = oklab_pixels[idx];
    }

    let mut counts = vec![0u32; k];
    let batch_size = 4096;
    let samples = n_pixels.min(512 * 512); // Sample cap to prevent infinite loops on massive inputs
    let noise_spread = dither_level * 0.1; // Reduced multiplier due to tight Oklab ranges

    // 3. MacQueen iterations
    for _ in 0..(samples / batch_size) {
        // Collect our batch indices
        let mut batch_indices = vec![0usize; batch_size];
        for idx in &mut batch_indices {
            *idx = (xorshift() as usize) % n_pixels;
        }

        // Step 1: In parallel, sample colors and find their nearest current centroid
        let assignments: Vec<usize> = batch_indices
            .par_iter()
            .map(|&idx| {
                let mut p = oklab_pixels[idx];

                // Blue noise jitter injection to assist with structural dispersion
                if noise_spread > 0.0 {
                    let x = (idx % width) as u32;
                    let y = (idx / width) as u32;
                    let noise_u8 = get_noise(x, y);
                    let noise = (noise_u8 as f32 / 255.0) - 0.5;
                    let jitter = noise * noise_spread;
                    p.l = (p.l + jitter).clamp(0.0, 1.0);
                    p.a += jitter;
                    p.b += jitter;
                }

                let mut min_dist = f32::MAX;
                let mut best_c = 0;
                for (ci, c) in centroids.iter().enumerate() {
                    let dl = p.l - c.l;
                    let da = p.a - c.a;
                    let db = p.b - c.b;
                    let dist = dl * dl + da * da + db * db;
                    if dist < min_dist {
                        min_dist = dist;
                        best_c = ci;
                    }
                }
                best_c
            })
            .collect();

        // Step 2: Sequentially update the original centroids based on assignments
        for (i, &idx) in batch_indices.iter().enumerate() {
            let c_idx = assignments[i];
            let mut p = oklab_pixels[idx];

            if noise_spread > 0.0 {
                let x = (idx % width) as u32;
                let y = (idx / width) as u32;
                let noise_u8 = get_noise(x, y);
                let noise = (noise_u8 as f32 / 255.0) - 0.5;
                let jitter = noise * noise_spread;
                p.l = (p.l + jitter).clamp(0.0, 1.0);
                p.a += jitter;
                p.b += jitter;
            }

            counts[c_idx] += 1;
            // Adaptive learning rate: 1.0 / sqrt(count)
            let rate = 1.0 / (counts[c_idx] as f32).sqrt();

            centroids[c_idx].l += rate * (p.l - centroids[c_idx].l);
            centroids[c_idx].a += rate * (p.a - centroids[c_idx].a);
            centroids[c_idx].b += rate * (p.b - centroids[c_idx].b);
        }
    }

    // 4. Final mapping back to sRGB [u8; 3]
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

/// K-means++ clustering in Oklab with Oklab Euclidean distance (used when bias > 0.0).
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

    if rgb_pixels.is_empty() {
        return vec![[0, 0, 0]; k];
    }

    let n_pixels = rgb_pixels.len();

    // ── Color histogram deduplication ───────────────────────────────────────
    // Typical photos have N pixels but only a small fraction of distinct RGB
    // triples. Collapsing duplicates into a single representative each cuts
    // the sRGB→Oklab conversion, the k-means++ distance updates, and the
    // 20-iteration nearest-centroid search from O(N·k) to O(U·k), where U is
    // the number of unique colors — often 10–100× smaller than N.
    //
    // FP determinism is preserved by keeping every weighted sum, threshold
    // walk, and accumulation pass iterating over the ORIGINAL pixel sequence
    // (via `pixel_to_unique` index lookups). Duplicate pixels share identical
    // Oklab values and identical per-pixel weights, so the resulting floating
    // point arithmetic is term-for-term identical to the pre-dedup version —
    // just with the expensive distance math cached once per unique color.
    let mut indexed_keys: Vec<(u32, u32)> = rgb_pixels
        .par_iter()
        .enumerate()
        .map(|(i, p)| {
            let key = ((p[0] as u32) << 16) | ((p[1] as u32) << 8) | (p[2] as u32);
            (key, i as u32)
        })
        .collect();
    indexed_keys.par_sort_unstable_by_key(|&(k, _)| k);

    let mut unique_rgb: Vec<[u8; 3]> = Vec::with_capacity(n_pixels / 4 + 1);
    let mut pixel_to_unique: Vec<u32> = vec![0u32; n_pixels];
    {
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
    }
    drop(indexed_keys);
    let n_unique = unique_rgb.len();

    // All heavy math (Oklab conversion, chroma, distances) now runs over
    // unique colors only. `oklab_pixels` is indexed by unique color id;
    // use `pixel_to_unique[i]` to map from an original pixel index.
    let oklab_pixels: Vec<Oklab> = unique_rgb.par_iter().map(|&p| srgb_to_oklab(p)).collect();

    // 1. Per-pixel weights: uniform base with optional linear chroma boost.
    //    sat_bias = 1.0 acts as the neutral floor (all pixels get a weight of 1.0),
    //    relying purely on `oklch_weighted_dist` to preserve vibrant colors.
    //    Values > 1.0 (e.g., 4.0) apply a gentle linear multiplier to
    //    artificially inflate the importance of highly saturated pixels.
    let chroma_multiplier = (sat_bias - 1.0).max(0.0);

    let chromas: Vec<f32> = oklab_pixels
        .iter()
        .map(|p| (p.a * p.a + p.b * p.b).sqrt())
        .collect();

    let mut weights: Vec<f32> = chromas
        .iter()
        .map(|&c| 1.0 + (c * chroma_multiplier))
        .collect();

    // 2. Histogram equalization across lightness and chroma.
    //    Two independent 16-bin histograms are computed, and their equalization
    //    factors are blended via lc_priority: 0.0 = lightness only, 1.0 = chroma
    //    only, 0.5 = both equally. This is what preserves dark-area detail —
    //    underrepresented lightness bins get boosted so k-means allocates
    //    palette slots to them proportionally.
    //
    //    Bin accumulation walks the ORIGINAL pixel order through pixel_to_unique
    //    so that the summed weights match the pre-dedup FP sequence term-for-term.
    const LC_BINS: usize = 16;

    // Equalization strength scales with light_boost
    let eq_power = (0.5_f32).max(1.0 - 0.5 / light_boost.max(0.1));

    // --- Lightness histogram ---
    let mut l_bin_weights = [0.0f32; LC_BINS];
    for i in 0..n_pixels {
        let u = pixel_to_unique[i] as usize;
        let bin = ((oklab_pixels[u].l * LC_BINS as f32) as usize).min(LC_BINS - 1);
        l_bin_weights[bin] += weights[u];
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
    for i in 0..n_pixels {
        let u = pixel_to_unique[i] as usize;
        let bin = ((chromas[u] * chroma_scale) as usize).min(LC_BINS - 1);
        c_bin_weights[bin] += weights[u];
    }
    let active_c = c_bin_weights.iter().filter(|&&w| w > 0.0).count() as f32;
    let avg_c = if active_c > 0.0 {
        c_bin_weights.iter().sum::<f32>() / active_c
    } else {
        1.0
    };

    // --- Apply blended equalization ---
    //   Equalization factors depend only on the pixel's Oklab value, so they
    //   are identical across duplicates. Applying them to the per-unique
    //   weights array therefore produces the same per-pixel weight the
    //   pre-dedup code would have produced for every original pixel.
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

    // 4. K-Means++ initialization
    //    Initial fallback value matches the pre-dedup first-pixel Oklab
    //    (the Oklab of rgb_pixels[0]), NOT unique index 0 which is now
    //    the lowest-keyed color after sorting.
    let first_pixel_oklab = oklab_pixels[pixel_to_unique[0] as usize];
    let mut centroids = vec![first_pixel_oklab; k];
    let mut min_dists = vec![f32::MAX; n_unique]; // Cache nearest centroid distance (per unique color)

    let mut rng_state: u64 = 0x5EED_C0DE_1234_5678;
    let mut xorshift = || -> u64 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        rng_state
    };

    // Initial weight sum walks the original pixel order so the
    // left-associated FP partial-sum sequence matches the pre-dedup
    // `weights.iter().sum()` term-for-term.
    let total_w: f32 = (0..n_pixels)
        .map(|i| weights[pixel_to_unique[i] as usize])
        .sum();
    let mut threshold = (xorshift() as f32 / u64::MAX as f32) * total_w;
    for i in 0..n_pixels {
        let u = pixel_to_unique[i] as usize;
        threshold -= weights[u];
        if threshold <= 0.0 {
            centroids[0] = oklab_pixels[u];
            break;
        }
    }

    // Initial pass: populate min_dists for the first centroid (over unique only).
    min_dists.par_iter_mut().enumerate().for_each(|(i, d)| {
        *d = oklch_weighted_dist(oklab_pixels[i], centroids[0]);
    });

    for ki in 1..k {
        // Calculate the weighted sum using cached min_dists.
        // Deterministic parallel sum: fixed-size chunks are summed in
        // parallel, then the chunk totals are combined sequentially in
        // index order. This avoids FP non-associativity jitter while
        // staying parallel for large pixel counts.
        //
        // Chunks walk ORIGINAL pixels (via pixel_to_unique) so the
        // summation order is identical to the pre-dedup version.
        const CHUNK_W: usize = 4096;
        let n_chunks_w = n_pixels.div_ceil(CHUNK_W);
        let total: f32 = (0..n_chunks_w)
            .into_par_iter()
            .map(|chunk_idx| {
                let start = chunk_idx * CHUNK_W;
                let end = (start + CHUNK_W).min(n_pixels);
                let mut s = 0.0f32;
                for i in start..end {
                    let u = pixel_to_unique[i] as usize;
                    s += min_dists[u] * weights[u];
                }
                s
            })
            .collect::<Vec<f32>>()
            .iter()
            .sum();

        if total <= 0.0 {
            // Fallback picks a random ORIGINAL pixel (pre-dedup modulus range),
            // then dereferences to its unique Oklab — identical semantics.
            let idx = (xorshift() as usize) % n_pixels;
            centroids[ki] = oklab_pixels[pixel_to_unique[idx] as usize];
        } else {
            let mut target = (xorshift() as f32 / u64::MAX as f32) * total;
            for i in 0..n_pixels {
                let u = pixel_to_unique[i] as usize;
                target -= min_dists[u] * weights[u];
                if target <= 0.0 {
                    centroids[ki] = oklab_pixels[u];
                    break;
                }
            }
        }

        let new_c = centroids[ki];

        // Update cached distances, ONLY checking against the newly added centroid
        min_dists.par_iter_mut().enumerate().for_each(|(i, d)| {
            let dist = oklch_weighted_dist(oklab_pixels[i], new_c);
            if dist < *d {
                *d = dist;
            }
        });
    }

    // 5. K-Means iterations
    for _ in 0..20 {
        // Extract slice so the compiler doesn't insert bounds checks in the inner loop
        let cd_slice = &centroids[..];

        // Step A: nearest-centroid search runs per UNIQUE color (O(U·k)
        // instead of O(N·k)). This is the dominant cost of k-means and the
        // single biggest win from dedup. The scalar < comparison preserves
        // first-index tie-breaking, so the selected best_idx is identical to
        // what the pre-dedup per-pixel loop would have chosen for every
        // duplicate of this color.
        // Assignment uses the same oklch_weighted_dist as k-means++ init and
        // the final mapping pass — one metric end-to-end keeps centroids
        // out of the "hue-distant but plain-Euclidean-close" traps that
        // previously left them unclaimed and drifting stale.
        let best_per_unique: Vec<u32> = (0..n_unique)
            .into_par_iter()
            .map(|u| {
                let p = oklab_pixels[u];

                let mut min_dist = f32::MAX;
                let mut best_idx = 0u32;
                for (ci, cd) in cd_slice.iter().enumerate() {
                    let dist_sq = oklch_weighted_dist(p, *cd);

                    if dist_sq < min_dist {
                        min_dist = dist_sq;
                        best_idx = ci as u32;
                    }
                }
                best_idx
            })
            .collect();

        // Step B: deterministic parallel accumulation over ORIGINAL pixels.
        // Fixed-size chunks are processed in parallel and collected in index
        // order, so the final sequential reduction always combines partial
        // sums identically across runs. This walks N pixels (not U) so the
        // FP sequence matches the pre-dedup implementation exactly; the only
        // per-pixel work here is three lookups and four FMAs.
        const CHUNK: usize = 4096;
        let n_chunks = n_pixels.div_ceil(CHUNK);
        let partials: Vec<(Vec<(f32, f32, f32)>, Vec<f32>)> = (0..n_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let start = chunk_idx * CHUNK;
                let end = (start + CHUNK).min(n_pixels);
                let mut sums = vec![(0.0f32, 0.0f32, 0.0f32); k];
                let mut counts = vec![0.0f32; k];
                for i in start..end {
                    let u = pixel_to_unique[i] as usize;
                    let p = &oklab_pixels[u];
                    let w = weights[u];
                    let best_idx = best_per_unique[u] as usize;

                    // Hoist pixel properties out of the centroid loop
                    let p_l = p.l;
                    let p_a = p.a;
                    let p_b = p.b;

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
                let new_ok = Oklab {
                    l: sums[i].0 / counts[i],
                    a: sums[i].1 / counts[i],
                    b: sums[i].2 / counts[i],
                };

                let shift = oklch_weighted_dist(centroids[i], new_ok);
                max_shift = max_shift.max(shift);
                centroids[i] = new_ok;
            }
        }

        // Empty-cluster reseed. Any centroid with counts == 0 got no pixels
        // assigned this iteration, so the update loop above left it at its
        // stale position. In the mapping pass that stale centroid can still
        // be oklch-nearest to some pixels and show up as a stray hue. Reseed
        // each empty centroid to the unique colour currently furthest from
        // any filled centroid — the same heuristic k-means++ uses for init.
        let mut reseeded = false;
        for empty_idx in 0..k {
            if counts[empty_idx] > 0.0 {
                continue;
            }
            // Parallel per-unique "distance to nearest filled centroid", then
            // sequential argmax for deterministic tie-breaking. Empty
            // centroids are handled one at a time so that each subsequent
            // reseed sees the previous one as "filled" and avoids landing
            // every empty slot on the same outlier.
            let min_d_per_u: Vec<f32> = (0..n_unique)
                .into_par_iter()
                .map(|u| {
                    let p = oklab_pixels[u];
                    let mut min_d = f32::MAX;
                    for (ci, cd) in centroids.iter().enumerate() {
                        if counts[ci] <= 0.0 {
                            continue;
                        }
                        let d = oklch_weighted_dist(p, *cd);
                        if d < min_d {
                            min_d = d;
                        }
                    }
                    min_d
                })
                .collect();

            let (best_u, _) =
                min_d_per_u
                    .iter()
                    .enumerate()
                    .fold(
                        (0usize, f32::MIN),
                        |(bu, bd), (u, &d)| if d > bd { (u, d) } else { (bu, bd) },
                    );

            centroids[empty_idx] = oklab_pixels[best_u];
            counts[empty_idx] = 1.0; // mark filled for later reseeds this pass
            reseeded = true;
        }
        if reseeded {
            // Force another iteration so reseeded centroids actually get
            // pixels assigned to them before the convergence test can fire.
            max_shift = max_shift.max(1.0);
        }

        if max_shift < 1e-4 {
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

#[inline(always)]
fn oklch_weighted_dist(p: Oklab, c: Oklab) -> f32 {
    let p_c = (p.a * p.a + p.b * p.b).sqrt();
    let c_c = (c.a * c.a + c.b * c.b).sqrt();

    // 1. Lightness difference
    let dl = p.l - c.l;

    // 2. Chroma difference
    let dc = p_c - c_c;

    // 3. Hue difference (chord length trick to avoid expensive atan2)
    let da = p.a - c.a;
    let db = p.b - c.b;
    let dh_sq = (da * da + db * db - dc * dc).max(0.0);

    // Perceptual Weights:
    // 1.0 for Lightness, 4.0 for Chroma, 4.0 for Hue.
    // This forces the algorithm to preserve vibrant gradients instead
    // of settling for mathematically safe but pale averages.
    (dl * dl * 1.0) + (dc * dc * 4.0) + (dh_sq * 4.0)
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
