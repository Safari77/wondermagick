use crate::{arg_parse_err::ArgParseErr, error::MagickError, image::Image};
use image::{DynamicImage, RgbaImage};
use oklab::{oklab_to_srgb, srgb_to_oklab, Oklab, Rgb};
use rayon::prelude::*;

/// One stop in a gradient.
#[derive(Debug, Clone, PartialEq)]
pub struct GradientStop {
    /// Position in [0.0, 1.0]
    pub pos: f64,
    /// Straight (non-premultiplied) RGBA
    pub color: [u8; 4],
}

/// A coordinate for gradient centers, either a ratio [0.0, 1.0] or absolute pixels.
#[derive(Debug, Clone, PartialEq)]
pub enum Coord {
    Ratio(f64),
    Pixels(f64),
}

/// What to fill the canvas with.
#[derive(Debug, Clone, PartialEq)]
pub enum CanvasSpec {
    /// Solid RGBA fill
    Solid([u8; 4]),
    /// Linear gradient at an arbitrary angle (in degrees).
    /// 0 deg = left-to-right, 90 deg = top-to-bottom (angle increases clockwise
    /// in screen coordinates).
    Linear {
        angle_deg: f64,
        stops: Vec<GradientStop>,
    },
    /// Radial gradient. `center_x` and `center_y` define the center. 0 at center, 1 at the farthest corner.
    Radial {
        center_x: Coord,
        center_y: Coord,
        stops: Vec<GradientStop>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct CanvasConfig {
    /// If `Some`, create a brand new image of this size, discarding any input.
    /// If `None`, reuse the dimensions of the current image and only overwrite
    /// its pixels. Use `size:WIDTHxHEIGHT` at the start of the spec to set this.
    pub size: Option<(u32, u32)>,
    pub spec: CanvasSpec,
}

impl CanvasConfig {
    /// Parse a canvas spec from a comma-separated string.
    ///
    /// Syntax:
    ///   [size:WxH,]solid,COLOR
    ///   [size:WxH,]linear,ANGLE_DEG,STOP1,STOP2[,STOP3...]
    ///   [size:WxH,]radial,[pos:x,y,]STOP1,STOP2[,STOP3...]
    ///
    /// When `size:WxH` is omitted, the canvas operation overwrites the current
    /// image's pixels while keeping its dimensions. When `size:WxH` is given,
    /// the current image is replaced with a new one of the requested size --
    /// useful when canvas is the first operation in the pipeline and no input
    /// image was loaded from disk.
    ///
    /// COLOR: hex, one of `#RGB`, `#RGBA`, `#RRGGBB`, `#RRGGBBAA`.
    ///
    /// STOP: either `POS:COLOR` (e.g. `0.5:#ff0000`), or just `COLOR` in which
    ///       case positions are evenly distributed in [0, 1]. Positional and
    ///       non-positional stops cannot be mixed in one spec.
    ///
    /// ANGLE_DEG: fractional degrees, e.g. `16.514`.
    pub fn parse_arg(s: &str) -> Result<Self, ArgParseErr> {
        let s = s.trim();
        let parts: Vec<&str> = s.split(',').map(|p| p.trim()).collect();
        if parts.is_empty() || parts[0].is_empty() {
            return Err(ArgParseErr::with_msg("canvas: empty argument"));
        }

        // Optional 'size:WxH' prefix must be the first comma-separated token.
        let (size, remaining): (Option<(u32, u32)>, &[&str]) =
            if let Some(size_str) = parts[0].strip_prefix("size:") {
                (Some(parse_size(size_str)?), &parts[1..])
            } else {
                (None, &parts[..])
            };

        if remaining.is_empty() {
            return Err(ArgParseErr::with_msg(
                "canvas: missing type (expected 'solid', 'linear', or 'radial')",
            ));
        }

        let kind = remaining[0].to_lowercase();

        let spec = match kind.as_str() {
            "solid" => {
                if remaining.len() != 2 {
                    return Err(ArgParseErr::with_msg(
                        "canvas solid: expected '[size:WxH,]solid,COLOR'",
                    ));
                }
                CanvasSpec::Solid(parse_color(remaining[1])?)
            }
            "linear" => {
                if remaining.len() < 4 {
                    return Err(ArgParseErr::with_msg(
                        "canvas linear: expected '[size:WxH,]linear,ANGLE,STOP1,STOP2[,...]' \
                         (at least 2 stops)",
                    ));
                }
                let angle_deg = remaining[1].parse::<f64>().map_err(|_| {
                    ArgParseErr::with_msg(
                        "canvas linear: invalid angle (expected a number in degrees)",
                    )
                })?;
                if !angle_deg.is_finite() {
                    return Err(ArgParseErr::with_msg("canvas linear: angle must be finite"));
                }
                let stops = parse_stops(&remaining[2..])?;
                CanvasSpec::Linear { angle_deg, stops }
            }
            "radial" => {
                let mut center_x = Coord::Ratio(0.5);
                let mut center_y = Coord::Ratio(0.5);
                let mut stops_start = 1;

                // Check for pos:x,y (which spans two comma-separated tokens: "pos:x" and "y")
                if remaining.len() >= 3 && remaining[1].starts_with("pos:") {
                    let x_str = remaining[1].strip_prefix("pos:").unwrap();
                    let y_str = remaining[2];
                    if let (Some(cx), Some(cy)) = (parse_coord(x_str), parse_coord(y_str)) {
                        center_x = cx;
                        center_y = cy;
                        stops_start = 3;
                    } else {
                        return Err(ArgParseErr::with_msg(
                            "canvas radial: invalid pos specifier (expected pos:x,y)",
                        ));
                    }
                }

                if remaining.len() - stops_start < 2 {
                    return Err(ArgParseErr::with_msg(
                        "canvas radial: expected '[size:WxH,]radial,[pos:x,y,]STOP1,STOP2[,...]' \
                         (at least 2 stops)",
                    ));
                }
                let stops = parse_stops(&remaining[stops_start..])?;
                CanvasSpec::Radial {
                    center_x,
                    center_y,
                    stops,
                }
            }
            _ => {
                return Err(ArgParseErr::with_msg(
                    "canvas: type must be 'solid', 'linear', or 'radial'",
                ))
            }
        };

        Ok(Self { size, spec })
    }
}

fn parse_size(s: &str) -> Result<(u32, u32), ArgParseErr> {
    // accept both 'x' and 'X' as separator
    let mut it = s.splitn(2, |c| c == 'x' || c == 'X');
    let w_str = it
        .next()
        .ok_or_else(|| ArgParseErr::with_msg("canvas size: missing width"))?;
    let h_str = it.next().ok_or_else(|| {
        ArgParseErr::with_msg("canvas size: missing height (expected WIDTHxHEIGHT)")
    })?;
    let w: u32 = w_str
        .parse()
        .map_err(|_| ArgParseErr::with_msg("canvas size: invalid width"))?;
    let h: u32 = h_str
        .parse()
        .map_err(|_| ArgParseErr::with_msg("canvas size: invalid height"))?;
    if w == 0 || h == 0 {
        return Err(ArgParseErr::with_msg(
            "canvas size: width and height must be greater than 0",
        ));
    }
    // Guard against astronomically large allocations that would overflow usize.
    let bytes = (w as u64)
        .checked_mul(h as u64)
        .and_then(|p| p.checked_mul(4));
    if bytes.map_or(true, |b| b > (isize::MAX as u64)) {
        return Err(ArgParseErr::with_msg(
            "canvas size: dimensions are too large for this platform",
        ));
    }
    Ok((w, h))
}

fn hex_digit(c: u8) -> Result<u8, ArgParseErr> {
    match c {
        b'0'..=b'9' => Ok(c - b'0'),
        b'a'..=b'f' => Ok(c - b'a' + 10),
        b'A'..=b'F' => Ok(c - b'A' + 10),
        _ => Err(ArgParseErr::with_msg("canvas color: invalid hex digit")),
    }
}

fn hex_byte(hi: u8, lo: u8) -> Result<u8, ArgParseErr> {
    Ok((hex_digit(hi)? << 4) | hex_digit(lo)?)
}

fn parse_coord(s: &str) -> Option<Coord> {
    if let Some(px_str) = s.strip_suffix("px") {
        let px: f64 = px_str.parse().ok()?;
        if px >= 0.0 {
            return Some(Coord::Pixels(px));
        }
    } else if let Ok(ratio) = s.parse::<f64>() {
        if (0.0..=1.0).contains(&ratio) {
            return Some(Coord::Ratio(ratio));
        }
    }
    None
}

fn parse_color(s: &str) -> Result<[u8; 4], ArgParseErr> {
    let s = s.trim();
    let hex = s.strip_prefix('#').ok_or_else(|| {
        ArgParseErr::with_msg("canvas color: must be hex like #RGB, #RGBA, #RRGGBB, or #RRGGBBAA")
    })?;
    let b = hex.as_bytes();
    match b.len() {
        3 => {
            // #RGB
            let r = hex_digit(b[0])?;
            let g = hex_digit(b[1])?;
            let bl = hex_digit(b[2])?;
            Ok([r * 0x11, g * 0x11, bl * 0x11, 0xFF])
        }
        4 => {
            // #RGBA
            let r = hex_digit(b[0])?;
            let g = hex_digit(b[1])?;
            let bl = hex_digit(b[2])?;
            let a = hex_digit(b[3])?;
            Ok([r * 0x11, g * 0x11, bl * 0x11, a * 0x11])
        }
        6 => {
            // #RRGGBB
            Ok([
                hex_byte(b[0], b[1])?,
                hex_byte(b[2], b[3])?,
                hex_byte(b[4], b[5])?,
                0xFF,
            ])
        }
        8 => {
            // #RRGGBBAA
            Ok([
                hex_byte(b[0], b[1])?,
                hex_byte(b[2], b[3])?,
                hex_byte(b[4], b[5])?,
                hex_byte(b[6], b[7])?,
            ])
        }
        _ => Err(ArgParseErr::with_msg(
            "canvas color: expected #RGB, #RGBA, #RRGGBB, or #RRGGBBAA",
        )),
    }
}

fn parse_stops(tokens: &[&str]) -> Result<Vec<GradientStop>, ArgParseErr> {
    if tokens.len() < 2 {
        return Err(ArgParseErr::with_msg(
            "canvas gradient: at least 2 stops required",
        ));
    }

    let any_positional = tokens.iter().any(|t| t.contains(':'));
    let all_positional = tokens.iter().all(|t| t.contains(':'));
    if any_positional && !all_positional {
        return Err(ArgParseErr::with_msg(
            "canvas gradient: either every stop must have a POS:COLOR form, or none (mixing not allowed)",
        ));
    }

    // The explicit `-> Result<GradientStop, ArgParseErr>` return type on these
    // closures is needed because `ArgParseErr` has a `From<ParseFloatError>`
    // impl, which makes the `?` operator's target error type ambiguous
    // (E0282 / E0283 otherwise).
    let mut stops: Vec<GradientStop> = if all_positional {
        tokens
            .iter()
            .map(|t| -> Result<GradientStop, ArgParseErr> {
                let mut it = t.splitn(2, ':');
                let pos_s = it.next().unwrap();
                let col_s = it.next().ok_or_else(|| {
                    ArgParseErr::with_msg("canvas gradient stop: missing color after ':'")
                })?;
                let pos: f64 = pos_s.trim().parse().map_err(|_| {
                    ArgParseErr::with_msg(
                        "canvas gradient stop: invalid position (expected number in [0,1])",
                    )
                })?;
                if !pos.is_finite() || !(0.0..=1.0).contains(&pos) {
                    return Err(ArgParseErr::with_msg(
                        "canvas gradient stop: position must be a finite number in [0.0, 1.0]",
                    ));
                }
                let color = parse_color(col_s)?;
                Ok(GradientStop { pos, color })
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        // Auto-distribute positions evenly across [0, 1].
        let n = tokens.len();
        tokens
            .iter()
            .enumerate()
            .map(|(i, t)| -> Result<GradientStop, ArgParseErr> {
                let pos = i as f64 / (n - 1) as f64; // n >= 2 ensured above
                let color = parse_color(t)?;
                Ok(GradientStop { pos, color })
            })
            .collect::<Result<Vec<_>, _>>()?
    };

    // Stable sort by position so equal positions preserve input order.
    stops.sort_by(|a, b| {
        a.pos
            .partial_cmp(&b.pos)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(stops)
}

/// A gradient pre-converted to Oklab for fast per-pixel evaluation.
/// Interpolating in Oklab gives perceptually uniform, visually pleasing
/// transitions (no muddy midpoints between complementary hues).
struct PreparedGradient {
    /// (position, oklab color, straight alpha)
    stops: Vec<(f64, Oklab, u8)>,
}

impl PreparedGradient {
    fn new(stops: &[GradientStop]) -> Self {
        let prepared = stops
            .iter()
            .map(|s| {
                let rgb = Rgb {
                    r: s.color[0],
                    g: s.color[1],
                    b: s.color[2],
                };
                (s.pos, srgb_to_oklab(rgb), s.color[3])
            })
            .collect();
        Self { stops: prepared }
    }

    #[inline]
    fn eval(&self, t: f64) -> [u8; 4] {
        if self.stops.is_empty() {
            return [0, 0, 0, 0xFF];
        }
        // Clamp-to-edge behaviour (t outside [first_pos, last_pos] -> edge color).
        let first = &self.stops[0];
        if self.stops.len() == 1 || t <= first.0 {
            let rgb = oklab_to_srgb(first.1);
            return [rgb.r, rgb.g, rgb.b, first.2];
        }
        let last = self.stops.last().unwrap();
        if t >= last.0 {
            let rgb = oklab_to_srgb(last.1);
            return [rgb.r, rgb.g, rgb.b, last.2];
        }

        // Locate the bracketing pair: last stop with pos <= t, and the next one.
        let i = self.stops.partition_point(|s| s.0 <= t).saturating_sub(1);
        let (pos_lo, lab_lo, a_lo) = self.stops[i];
        let (pos_hi, lab_hi, a_hi) = self.stops[i + 1];

        let span = pos_hi - pos_lo;
        let local_t = if span > 0.0 { (t - pos_lo) / span } else { 0.0 };
        let tf = local_t as f32;

        let mixed = Oklab {
            l: lab_lo.l + (lab_hi.l - lab_lo.l) * tf,
            a: lab_lo.a + (lab_hi.a - lab_lo.a) * tf,
            b: lab_lo.b + (lab_hi.b - lab_lo.b) * tf,
        };
        let rgb = oklab_to_srgb(mixed);
        // Alpha is interpolated linearly in straight-alpha space.
        let alpha = (a_lo as f64 + (a_hi as f64 - a_lo as f64) * local_t)
            .round()
            .clamp(0.0, 255.0) as u8;
        [rgb.r, rgb.g, rgb.b, alpha]
    }
}

pub fn canvas(image: &mut Image, config: &CanvasConfig) -> Result<(), MagickError> {
    // 1. Determine size AND whether we should composite or replace
    let (_width, _height, is_overlay) = match config.size {
        Some((w, h)) => (w, h, false),
        None => {
            let w = image.pixels.width();
            let h = image.pixels.height();
            if w == 0 || h == 0 {
                return Err(crate::wm_err!(
                    "canvas: current image has zero dimensions; use 'size:WIDTHxHEIGHT,...' to create a new canvas"
                ));
            }
            (w, h, true)
        }
    };

    // Resolve target dimensions. With `size:WxH` the canvas creates a brand
    // new image; without it, we overwrite the pixels of the current image
    // while keeping its dimensions. The latter is useful when canvas is
    // chained AFTER another operation that already produced an image.
    let (width, height) = match config.size {
        Some((w, h)) => (w, h),
        None => {
            // Use the inherent `width()`/`height()` on DynamicImage so we don't
            // have to bring `image::GenericImageView` into scope.
            let w = image.pixels.width();
            let h = image.pixels.height();
            if w == 0 || h == 0 {
                return Err(crate::wm_err!(
                    "canvas: current image has zero dimensions; use 'size:WIDTHxHEIGHT,...' to create a new canvas"
                ));
            }
            (w, h)
        }
    };

    // checked in parse_size for the size: path, but verify again for defense
    // in depth (config could be built directly by callers, or inherited from
    // a very large input image).
    let buf_len = (width as usize)
        .checked_mul(height as usize)
        .and_then(|p| p.checked_mul(4))
        .ok_or_else(|| crate::wm_err!("canvas: image dimensions overflow usize"))?;

    let mut buf: Vec<u8> = vec![0; buf_len];
    let row_bytes = width as usize * 4;

    match &config.spec {
        CanvasSpec::Solid(c) => {
            // Parallel fill. chunks_exact_mut of size 4 is faster than .copy_from_slice on
            // the whole row because the optimiser turns it into a memset-friendly loop.
            buf.par_chunks_mut(row_bytes).for_each(|row| {
                for px in row.chunks_exact_mut(4) {
                    px.copy_from_slice(c);
                }
            });
        }
        CanvasSpec::Linear { angle_deg, stops } => {
            let prepared = PreparedGradient::new(stops);

            // Gradient direction (cos, sin) in image coordinates (y grows DOWN):
            //   angle 0   deg -> (+1,  0) : left -> right
            //   angle 90  deg -> ( 0, +1) : top  -> bottom (clockwise when viewed)
            //   angle 180 deg -> (-1,  0) : right -> left
            //   angle 270 deg -> ( 0, -1) : bottom -> top
            let theta = angle_deg.to_radians();
            let cos_t = theta.cos();
            let sin_t = theta.sin();

            // Project every corner on the gradient axis and use the largest
            // absolute projection so the gradient spans the whole visible area,
            // regardless of rotation. Using `(W-1)/2` gives perfect end-color
            // coverage at the corner pixels. f64 gives sub-pixel accuracy even
            // for angles like 16.514 deg.
            let cx = (width as f64 - 1.0) / 2.0;
            let cy = (height as f64 - 1.0) / 2.0;
            let max_proj = cx * cos_t.abs() + cy * sin_t.abs();
            let total_span = 2.0 * max_proj;

            buf.par_chunks_mut(row_bytes)
                .enumerate()
                .for_each(|(y, row)| {
                    let dy = y as f64 - cy;
                    let dy_sin = dy * sin_t;
                    for (x, px) in row.chunks_exact_mut(4).enumerate() {
                        let dx = x as f64 - cx;
                        let proj = dx * cos_t + dy_sin;
                        let t = if total_span > 0.0 {
                            ((proj + max_proj) / total_span).clamp(0.0, 1.0)
                        } else {
                            0.0
                        };
                        let c = prepared.eval(t);
                        px.copy_from_slice(&c);
                    }
                });
        }
        CanvasSpec::Radial {
            center_x,
            center_y,
            stops,
        } => {
            let prepared = PreparedGradient::new(stops);

            let cx = match center_x {
                Coord::Ratio(r) => r * (width as f64 - 1.0),
                Coord::Pixels(p) => {
                    if *p < 0.0 || *p > width as f64 {
                        return Err(crate::wm_err!(
                            "canvas radial: xpos in px exceeds image width"
                        ));
                    }
                    *p
                }
            };
            let cy = match center_y {
                Coord::Ratio(r) => r * (height as f64 - 1.0),
                Coord::Pixels(p) => {
                    if *p < 0.0 || *p > height as f64 {
                        return Err(crate::wm_err!(
                            "canvas radial: ypos in px exceeds image height"
                        ));
                    }
                    *p
                }
            };

            // Calculate distance to the farthest corner from the chosen center
            let corners = [
                (0.0, 0.0),
                (width as f64 - 1.0, 0.0),
                (0.0, height as f64 - 1.0),
                (width as f64 - 1.0, height as f64 - 1.0),
            ];
            let max_r = corners
                .iter()
                .map(|&(px, py)| ((px - cx) * (px - cx) + (py - cy) * (py - cy)).sqrt())
                .fold(0.0_f64, |a, b| a.max(b));

            let inv_max_r = if max_r > 0.0 { 1.0 / max_r } else { 0.0 };

            buf.par_chunks_mut(row_bytes)
                .enumerate()
                .for_each(|(y, row)| {
                    let dy = y as f64 - cy;
                    let dy2 = dy * dy;
                    for (x, px) in row.chunks_exact_mut(4).enumerate() {
                        let dx = x as f64 - cx;
                        let r = (dx * dx + dy2).sqrt();
                        let t = (r * inv_max_r).clamp(0.0, 1.0);
                        let c = prepared.eval(t);
                        px.copy_from_slice(&c);
                    }
                });
        }
    }

    let out = RgbaImage::from_raw(width, height, buf).ok_or_else(|| {
        crate::wm_err!("canvas: failed to construct image buffer (dimensions too large?)")
    })?;

    if is_overlay {
        // We are mutating an existing image. Convert to RGBA and alpha-blend `out` OVER it.
        let mut base = image.pixels.to_rgba8();
        image::imageops::overlay(&mut base, &out, 0, 0);
        image.pixels = DynamicImage::ImageRgba8(base);
    } else {
        // We are generating a new image.
        image.pixels = DynamicImage::ImageRgba8(out);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_size_ok() {
        assert_eq!(parse_size("10x20").unwrap(), (10, 20));
        assert_eq!(parse_size("1X1").unwrap(), (1, 1));
    }

    #[test]
    fn parse_size_bad() {
        assert!(parse_size("0x10").is_err());
        assert!(parse_size("10x0").is_err());
        assert!(parse_size("abcx10").is_err());
        assert!(parse_size("10").is_err());
    }

    #[test]
    fn parse_color_hex_forms() {
        assert_eq!(parse_color("#f00").unwrap(), [0xFF, 0, 0, 0xFF]);
        assert_eq!(parse_color("#f008").unwrap(), [0xFF, 0, 0, 0x88]);
        assert_eq!(parse_color("#ff0000").unwrap(), [0xFF, 0, 0, 0xFF]);
        assert_eq!(parse_color("#ff000080").unwrap(), [0xFF, 0, 0, 0x80]);
    }

    #[test]
    fn parse_color_bad() {
        assert!(parse_color("ff0000").is_err()); // missing '#'
        assert!(parse_color("#zzz").is_err());
        assert!(parse_color("#12345").is_err()); // length 5 invalid
    }

    #[test]
    fn parse_stops_auto_positions() {
        let s = parse_stops(&["#ff0000", "#00ff00", "#0000ff"]).unwrap();
        assert_eq!(s.len(), 3);
        assert_eq!(s[0].pos, 0.0);
        assert!((s[1].pos - 0.5).abs() < 1e-12);
        assert_eq!(s[2].pos, 1.0);
    }

    #[test]
    fn parse_stops_explicit_positions() {
        let s = parse_stops(&["0:#ff0000", "0.25:#00ff00", "1:#0000ff"]).unwrap();
        assert_eq!(s[0].pos, 0.0);
        assert_eq!(s[1].pos, 0.25);
        assert_eq!(s[2].pos, 1.0);
    }

    #[test]
    fn parse_stops_mixed_positions_rejected() {
        assert!(parse_stops(&["0:#ff0000", "#00ff00"]).is_err());
    }

    #[test]
    fn parse_arg_solid_with_size() {
        let c = CanvasConfig::parse_arg("size:100x50,solid,#336699").unwrap();
        assert_eq!(c.size, Some((100, 50)));
        assert_eq!(c.spec, CanvasSpec::Solid([0x33, 0x66, 0x99, 0xFF]));
    }

    #[test]
    fn parse_arg_solid_without_size() {
        // `size:` is optional -- canvas inherits the current image's dimensions
        let c = CanvasConfig::parse_arg("solid,#336699").unwrap();
        assert_eq!(c.size, None);
        assert_eq!(c.spec, CanvasSpec::Solid([0x33, 0x66, 0x99, 0xFF]));
    }

    #[test]
    fn parse_arg_linear_fractional_angle() {
        let c = CanvasConfig::parse_arg("size:64x64,linear,16.514,#000000,#ffffff").unwrap();
        assert_eq!(c.size, Some((64, 64)));
        match c.spec {
            CanvasSpec::Linear { angle_deg, stops } => {
                assert!((angle_deg - 16.514).abs() < 1e-9);
                assert_eq!(stops.len(), 2);
            }
            _ => panic!("expected linear"),
        }
    }

    #[test]
    fn parse_arg_linear_without_size() {
        // linear gradient without size: uses current image dimensions
        let c = CanvasConfig::parse_arg("linear,45,#ff0000,#0000ff").unwrap();
        assert_eq!(c.size, None);
        assert!(matches!(c.spec, CanvasSpec::Linear { .. }));
    }

    #[test]
    fn parse_arg_radial_with_size() {
        let c = CanvasConfig::parse_arg("size:64x64,radial,0:#ffffff,1:#000000").unwrap();
        assert_eq!(c.size, Some((64, 64)));
        match c.spec {
            CanvasSpec::Radial {
                center_x,
                center_y,
                stops,
            } => {
                assert_eq!(center_x, Coord::Ratio(0.5));
                assert_eq!(center_y, Coord::Ratio(0.5));
                assert_eq!(stops.len(), 2);
            }
            _ => panic!("expected radial"),
        }
    }

    #[test]
    fn parse_arg_radial_without_size() {
        let c = CanvasConfig::parse_arg("radial,#ffffff,#000000").unwrap();
        assert_eq!(c.size, None);
        match c.spec {
            CanvasSpec::Radial {
                center_x,
                center_y,
                stops,
            } => {
                assert_eq!(center_x, Coord::Ratio(0.5));
                assert_eq!(center_y, Coord::Ratio(0.5));
                assert_eq!(stops.len(), 2);
            }
            _ => panic!("expected radial"),
        }
    }

    #[test]
    fn parse_arg_radial_with_coords_no_size() {
        let c = CanvasConfig::parse_arg("radial,pos:0.75,20px,#ffffff,#000000").unwrap();
        assert_eq!(c.size, None);
        match c.spec {
            CanvasSpec::Radial {
                center_x,
                center_y,
                stops,
            } => {
                assert_eq!(center_x, Coord::Ratio(0.75));
                assert_eq!(center_y, Coord::Pixels(20.0));
                assert_eq!(stops.len(), 2);
            }
            _ => panic!("expected radial"),
        }
    }

    #[test]
    fn parse_arg_radial_invalid_pos_rejected() {
        // Missing the y-coordinate; the parser reads `#ff00ff` as the y-coordinate and fails.
        assert!(CanvasConfig::parse_arg("radial,pos:0.75,#ff00ff,#000000").is_err());

        // Also fails if missing the y-coordinate and missing enough stops
        assert!(CanvasConfig::parse_arg("radial,pos:0.75,#ff00ff").is_err());

        // Fails with completely invalid coordinate formats
        assert!(CanvasConfig::parse_arg("radial,pos:abc,def,#ffffff,#000000").is_err());
    }

    #[test]
    fn parse_arg_missing_type() {
        assert!(CanvasConfig::parse_arg("size:10x10").is_err());
    }

    #[test]
    fn parse_arg_bad_type() {
        assert!(CanvasConfig::parse_arg("size:10x10,triangle,#ff0000").is_err());
    }

    #[test]
    fn eval_gradient_endpoints() {
        let pg = PreparedGradient::new(&[
            GradientStop {
                pos: 0.0,
                color: [255, 0, 0, 255],
            },
            GradientStop {
                pos: 1.0,
                color: [0, 0, 255, 255],
            },
        ]);
        let lo = pg.eval(0.0);
        let hi = pg.eval(1.0);
        // endpoints should round-trip through Oklab very close to original sRGB
        assert!(lo[0] > 240 && lo[1] < 15 && lo[2] < 15);
        assert!(hi[0] < 15 && hi[1] < 15 && hi[2] > 240);
        // alpha preserved exactly
        assert_eq!(lo[3], 255);
        assert_eq!(hi[3], 255);
    }
}
