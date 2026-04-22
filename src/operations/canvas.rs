use crate::{arg_parse_err::ArgParseErr, error::MagickError, image::Image};
use image::{DynamicImage, RgbaImage};
use kurbo::{CubicBez, ParamCurve, Point};
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

/// Defines the physical boundaries of a Coons Patch.
/// The curves must meet at the 4 corners:
/// top.p0 == left.p0
/// top.p3 == right.p0
/// bottom.p0 == left.p3
/// bottom.p3 == right.p3
pub struct CoonsPatch {
    pub top: CubicBez,
    pub bottom: CubicBez,
    pub left: CubicBez,
    pub right: CubicBez,
}

/// An easing function defined by a CSS-style cubic Bezier curve.
/// The curve is always anchored at (0,0) and (1,1).
#[derive(Debug, Clone, PartialEq)]
pub struct CssEasing {
    p1: (f64, f64),
    p2: (f64, f64),
}

impl CssEasing {
    /// Creates a new easing curve. Matches the CSS `cubic-bezier(x1, y1, x2, y2)`.
    pub fn new(x1: f64, y1: f64, x2: f64, y2: f64) -> Self {
        Self {
            p1: (x1, y1),
            p2: (x2, y2),
        }
    }

    /// Evaluates the eased progress for a given linear input `x` in [0.0, 1.0].
    #[inline]
    pub fn ease(&self, x: f64) -> f64 {
        let x = x.clamp(0.0, 1.0);
        let curve = CubicBez::new(
            Point::new(0.0, 0.0),
            Point::new(self.p1.0, self.p1.1),
            Point::new(self.p2.0, self.p2.1),
            Point::new(1.0, 1.0),
        );

        // Fast binary search to find the parameter `t` where curve.eval(t).x ≈ x
        let mut t_min = 0.0;
        let mut t_max = 1.0;
        let mut t = 0.5;

        // 12 iterations provide ~0.00024 precision, which is imperceptible in 8-bit color
        for _ in 0..12 {
            let current_x = curve.eval(t).x;
            if current_x < x {
                t_min = t;
            } else {
                t_max = t;
            }
            t = (t_min + t_max) * 0.5;
        }

        // Return the y value at the found t, clamped to valid color range
        curve.eval(t).y.clamp(0.0, 1.0)
    }
}

/// Interpolates 4 corner colors using Bilinear interpolation in Oklab space.
#[derive(Debug, Clone, PartialEq)]
pub struct MeshColors {
    pub tl: (Oklab, f32),
    pub tr: (Oklab, f32),
    pub bl: (Oklab, f32),
    pub br: (Oklab, f32),
}

impl MeshColors {
    #[inline]
    pub fn eval_color(&self, u: f64, v: f64) -> [u8; 4] {
        // Cast coordinates to f32 to match Oklab's internal precision
        let u = u as f32;
        let v = v as f32;

        let u_inv = 1.0 - u;
        let v_inv = 1.0 - v;

        let w00 = u_inv * v_inv;
        let w10 = u * v_inv;
        let w01 = u_inv * v;
        let w11 = u * v;

        // Blend Oklab color channels
        let mixed_oklab = Oklab {
            l: w00 * self.tl.0.l + w10 * self.tr.0.l + w01 * self.bl.0.l + w11 * self.br.0.l,
            a: w00 * self.tl.0.a + w10 * self.tr.0.a + w01 * self.bl.0.a + w11 * self.br.0.a,
            b: w00 * self.tl.0.b + w10 * self.tr.0.b + w01 * self.bl.0.b + w11 * self.br.0.b,
        };

        // Blend straight alpha channel
        let mixed_alpha = w00 * self.tl.1 + w10 * self.tr.1 + w01 * self.bl.1 + w11 * self.br.1;

        let rgb = oklab_to_srgb(mixed_oklab);

        // Return the RGB along with the calculated alpha
        [
            rgb.r,
            rgb.g,
            rgb.b,
            mixed_alpha.round().clamp(0.0, 255.0) as u8,
        ]
    }
}

impl CoonsPatch {
    /// Evaluates the geometric position inside the patch.
    /// `u` (horizontal) and `v` (vertical) are in [0.0, 1.0].
    #[inline]
    pub fn eval_position(&self, u: f64, v: f64) -> Point {
        let ct = self.top.eval(u);
        let cb = self.bottom.eval(u);
        let cl = self.left.eval(v);
        let cr = self.right.eval(v);

        let p00 = self.top.p0; // Top-Left
        let p10 = self.top.p3; // Top-Right
        let p01 = self.bottom.p0; // Bottom-Left
        let p11 = self.bottom.p3; // Bottom-Right

        // L_c(u, v): Linear interpolation between Top and Bottom curves
        let lc_x = (1.0 - v) * ct.x + v * cb.x;
        let lc_y = (1.0 - v) * ct.y + v * cb.y;

        // L_d(u, v): Linear interpolation between Left and Right curves
        let ld_x = (1.0 - u) * cl.x + u * cr.x;
        let ld_y = (1.0 - u) * cl.y + u * cr.y;

        // B(u, v): Bilinear interpolation of the 4 corners
        let u_inv = 1.0 - u;
        let v_inv = 1.0 - v;

        let b_x = u_inv * v_inv * p00.x + u * v_inv * p10.x + u_inv * v * p01.x + u * v * p11.x;

        let b_y = u_inv * v_inv * p00.y + u * v_inv * p10.y + u_inv * v * p01.y + u * v * p11.y;

        // Final Coons Patch formula
        Point::new(lc_x + ld_x - b_x, lc_y + ld_y - b_y)
    }

    /// Uses a Newton-Raphson solver to find the (u, v) coordinates for a given (x, y) pixel.
    /// Returns None if the pixel falls outside the boundaries of the curved patch.
    pub fn inverse_eval(&self, p: Point) -> Option<(f64, f64)> {
        let mut u = 0.5;
        let mut v = 0.5;
        let eps = 1e-4; // Finite difference step for the Jacobian

        for _ in 0..15 {
            // 15 iterations ensures solid convergence
            let current = self.eval_position(u, v);
            let err_x = current.x - p.x;
            let err_y = current.y - p.y;

            // If we are within half a pixel of the target, we've converged!
            if err_x.abs() < 0.5 && err_y.abs() < 0.5 {
                // Return clamped to prevent float slop right on the boundary
                return Some((u.clamp(0.0, 1.0), v.clamp(0.0, 1.0)));
            }

            // Approximate partial derivatives (Jacobian matrix)
            let pu = self.eval_position(u + eps, v);
            let pv = self.eval_position(u, v + eps);

            let du_x = (pu.x - current.x) / eps;
            let du_y = (pu.y - current.y) / eps;
            let dv_x = (pv.x - current.x) / eps;
            let dv_y = (pv.y - current.y) / eps;

            let det = du_x * dv_y - du_y * dv_x;
            if det.abs() < 1e-8 {
                return None;
            } // Singular matrix, fails to converge

            let step_u = (err_x * dv_y - err_y * dv_x) / det;
            let step_v = (err_y * du_x - err_x * du_y) / det;

            u -= step_u;
            v -= step_v;

            // If the solver wanders wildly outside the 0.0-1.0 UV space,
            // the pixel is outside the patch geometry.
            if u < -0.2 || u > 1.2 || v < -0.2 || v > 1.2 {
                return None;
            }
        }

        // Final fallback check
        if (0.0..=1.0).contains(&u) && (0.0..=1.0).contains(&v) {
            Some((u, v))
        } else {
            None
        }
    }
}

/// Interpolates 4 corner colors using Bilinear interpolation in Oklab space, preserving Alpha.
#[derive(Debug, Clone, PartialEq)]
pub struct PatchColors {
    pub top_left: (Oklab, f32),
    pub top_right: (Oklab, f32),
    pub bottom_left: (Oklab, f32),
    pub bottom_right: (Oklab, f32),
}

impl PatchColors {
    #[inline]
    pub fn eval_color(&self, u: f64, v: f64) -> [u8; 4] {
        // Cast coordinates to f32 to match Oklab's internal precision
        let u = u as f32;
        let v = v as f32;

        let u_inv = 1.0 - u;
        let v_inv = 1.0 - v;

        // Bilinear blend weights
        let w00 = u_inv * v_inv;
        let w10 = u * v_inv;
        let w01 = u_inv * v;
        let w11 = u * v;

        // Blend Oklab color channels
        let mixed_oklab = Oklab {
            l: w00 * self.top_left.0.l
                + w10 * self.top_right.0.l
                + w01 * self.bottom_left.0.l
                + w11 * self.bottom_right.0.l,
            a: w00 * self.top_left.0.a
                + w10 * self.top_right.0.a
                + w01 * self.bottom_left.0.a
                + w11 * self.bottom_right.0.a,
            b: w00 * self.top_left.0.b
                + w10 * self.top_right.0.b
                + w01 * self.bottom_left.0.b
                + w11 * self.bottom_right.0.b,
        };

        // Blend straight alpha channel
        let mixed_alpha = w00 * self.top_left.1
            + w10 * self.top_right.1
            + w01 * self.bottom_left.1
            + w11 * self.bottom_right.1;

        let rgb = oklab_to_srgb(mixed_oklab);

        // Return the RGB along with the calculated alpha
        [
            rgb.r,
            rgb.g,
            rgb.b,
            mixed_alpha.round().clamp(0.0, 255.0) as u8,
        ]
    }
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
        easing: Option<CssEasing>,
    },
    /// Radial gradient. `center_x` and `center_y` define the center. 0 at center, 1 at the farthest corner.
    Radial {
        center_x: Coord,
        center_y: Coord,
        stops: Vec<GradientStop>,
        easing: Option<CssEasing>,
    },
    /// A 4-corner bilinear mesh gradient.
    Mesh { colors: MeshColors },
    /// A Coons patch canvas with an auto-generated, interesting curved shape.
    /// Intended as a background layer: the 4 boundary curves bulge outside the
    /// canvas so every pixel is guaranteed to be covered. Both the shape and
    /// (optionally) the corner colors are generated from a deterministic PRNG,
    /// so output is reproducible via `seed:N`.
    Coons {
        /// PRNG seed. If `None`, one is derived from `getrandom` and reported
        /// on stderr so the output can be reproduced later.
        seed: Option<u64>,
        /// Corner colors. If `None`, a harmonious palette is generated from
        /// the same seed.
        colors: Option<PatchColors>,
        /// Optional CSS easing applied independently to u and v before the
        /// color lookup, giving non-linear gradient transitions inside the patch.
        easing: Option<CssEasing>,
        /// Optional transparency range `(min, max)` in `[0.0, 1.0]`. When set,
        /// pixel alpha is multiplied by `(1 - t)` where `t` is a bezier-derived
        /// scalar field that varies between `min` and `max` across the canvas.
        /// Pixels near the bezier "spine" curves get `t=min` (least transparent),
        /// pixels far from them get `t=max` (most transparent).
        transparency: Option<(f64, f64)>,
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
    ///   [size:WxH,]mesh,TL_COLOR,TR_COLOR,BL_COLOR,BR_COLOR
    ///   [size:WxH,]coons[,seed:N][,transparency:MIN-MAX][,TL_COLOR,TR_COLOR,BL_COLOR,BR_COLOR]
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

        // Helper to extract easing from remaining tokens, replacing commas with colons in syntax
        // Syntax: ease:x1:y1:x2:y2
        let mut easing = None;
        let mut filtered_remaining = Vec::new();
        for &token in remaining {
            if let Some(ease_str) = token.strip_prefix("ease:") {
                let pts: Result<Vec<f64>, _> = ease_str.split(':').map(|n| n.parse()).collect();
                if let Ok(p) = pts {
                    if p.len() == 4 {
                        easing = Some(CssEasing::new(p[0], p[1], p[2], p[3]));
                        continue;
                    }
                }
                return Err(ArgParseErr::with_msg(
                    "canvas: invalid easing format (expected ease:x1:y1:x2:y2)",
                ));
            }
            filtered_remaining.push(token);
        }

        let remaining = filtered_remaining; // Shadow with filtered list

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
                        "canvas linear: expected '[size:WxH,]linear,ANGLE,[ease:x:y:x:y,]STOP1,STOP2[,...]' \
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
                CanvasSpec::Linear {
                    angle_deg,
                    stops,
                    easing,
                }
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
                        "canvas radial: expected '[size:WxH,]radial,[pos:x,y,][ease:x:y:x:y,]STOP1,STOP2[,...]' \
                         (at least 2 stops)",
                    ));
                }
                let stops = parse_stops(&remaining[stops_start..])?;
                CanvasSpec::Radial {
                    center_x,
                    center_y,
                    stops,
                    easing,
                }
            }
            "mesh" => {
                if remaining.len() != 5 {
                    return Err(ArgParseErr::with_msg(
                        "canvas mesh: expected '[size:WxH,]mesh,TL_COLOR,TR_COLOR,BL_COLOR,BR_COLOR'",
                    ));
                }
                let c_tl = parse_color(remaining[1])?;
                let c_tr = parse_color(remaining[2])?;
                let c_bl = parse_color(remaining[3])?;
                let c_br = parse_color(remaining[4])?;

                let to_oklab = |c: [u8; 4]| {
                    (
                        srgb_to_oklab(Rgb {
                            r: c[0],
                            g: c[1],
                            b: c[2],
                        }),
                        c[3] as f32,
                    )
                };

                CanvasSpec::Mesh {
                    colors: MeshColors {
                        tl: to_oklab(c_tl),
                        tr: to_oklab(c_tr),
                        bl: to_oklab(c_bl),
                        br: to_oklab(c_br),
                    },
                }
            }
            "coons" => {
                // Coons syntax is flexible:
                //   coons                             -- fully random (shape + palette)
                //   coons,seed:N                      -- seeded random
                //   coons,TL,TR,BL,BR                 -- random shape, explicit colors
                //   coons,seed:N,TL,TR,BL,BR          -- fully deterministic
                //   coons,transparency:MIN-MAX        -- smooth bezier-based alpha field
                //   coons,transparency:X              -- uniform alpha multiplier (1-X)
                // Tokens `seed:`, `transparency:` and `ease:` may appear in any
                // order; colors, if given, must be exactly 4 hex colors.
                let mut seed: Option<u64> = None;
                let mut transparency: Option<(f64, f64)> = None;
                let mut color_tokens: Vec<&str> = Vec::new();

                for &token in &remaining[1..] {
                    if let Some(seed_str) = token.strip_prefix("seed:") {
                        if seed.is_some() {
                            return Err(ArgParseErr::with_msg(
                                "canvas coons: seed specified more than once",
                            ));
                        }
                        let s: u64 = seed_str.parse().map_err(|_| {
                            ArgParseErr::with_msg(
                                "canvas coons: invalid seed (expected non-negative integer fitting in u64)",
                            )
                        })?;
                        seed = Some(s);
                    } else if let Some(trans_str) = token.strip_prefix("transparency:") {
                        if transparency.is_some() {
                            return Err(ArgParseErr::with_msg(
                                "canvas coons: transparency specified more than once",
                            ));
                        }
                        // Accept `MIN-MAX` or a bare `X` (uniform).
                        let (t_min, t_max) = match trans_str.split_once('-') {
                            Some((min_s, max_s)) => {
                                let mi: f64 = min_s.trim().parse().map_err(|_| {
                                    ArgParseErr::with_msg(
                                        "canvas coons: invalid transparency min (expected number in [0,1])",
                                    )
                                })?;
                                let ma: f64 = max_s.trim().parse().map_err(|_| {
                                    ArgParseErr::with_msg(
                                        "canvas coons: invalid transparency max (expected number in [0,1])",
                                    )
                                })?;
                                (mi, ma)
                            }
                            None => {
                                let x: f64 = trans_str.trim().parse().map_err(|_| {
                                    ArgParseErr::with_msg(
                                        "canvas coons: invalid transparency (expected 'MIN-MAX' or a single number in [0,1])",
                                    )
                                })?;
                                (x, x)
                            }
                        };
                        if !t_min.is_finite() || !t_max.is_finite() {
                            return Err(ArgParseErr::with_msg(
                                "canvas coons: transparency values must be finite",
                            ));
                        }
                        if !(0.0..=1.0).contains(&t_min) || !(0.0..=1.0).contains(&t_max) {
                            return Err(ArgParseErr::with_msg(
                                "canvas coons: transparency values must be in [0.0, 1.0]",
                            ));
                        }
                        if t_min > t_max {
                            return Err(ArgParseErr::with_msg(
                                "canvas coons: transparency min must be <= max",
                            ));
                        }
                        transparency = Some((t_min, t_max));
                    } else {
                        color_tokens.push(token);
                    }
                }

                let colors = match color_tokens.len() {
                    0 => None,
                    4 => {
                        let c_tl = parse_color(color_tokens[0])?;
                        let c_tr = parse_color(color_tokens[1])?;
                        let c_bl = parse_color(color_tokens[2])?;
                        let c_br = parse_color(color_tokens[3])?;

                        let to_oklab = |c: [u8; 4]| {
                            (
                                srgb_to_oklab(Rgb {
                                    r: c[0],
                                    g: c[1],
                                    b: c[2],
                                }),
                                c[3] as f32,
                            )
                        };

                        Some(PatchColors {
                            top_left: to_oklab(c_tl),
                            top_right: to_oklab(c_tr),
                            bottom_left: to_oklab(c_bl),
                            bottom_right: to_oklab(c_br),
                        })
                    }
                    _ => {
                        return Err(ArgParseErr::with_msg(
                            "canvas coons: expected '[size:WxH,]coons[,seed:N][,transparency:MIN-MAX][,TL_COLOR,TR_COLOR,BL_COLOR,BR_COLOR]' \
                             (colors, if given, must be exactly 4)",
                        ));
                    }
                };

                CanvasSpec::Coons {
                    seed,
                    colors,
                    easing,
                    transparency,
                }
            }
            _ => {
                return Err(ArgParseErr::with_msg(
                    "canvas: type must be 'solid', 'linear', 'radial', 'mesh', or 'coons'",
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

/// Tiny deterministic PRNG (SplitMix64). Inlined so we don't need to pull in
/// `rand` just to draw a dozen random numbers per canvas. Quality is more than
/// enough for picking patch geometry and palette jitter.
/// Reference: https://prng.di.unimi.it/splitmix64.c
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform f64 in [0.0, 1.0).
    #[inline]
    fn next_f64(&mut self) -> f64 {
        // Top 53 bits fill an f64 mantissa exactly.
        (self.next_u64() >> 11) as f64 * (1.0_f64 / (1u64 << 53) as f64)
    }

    /// Uniform f64 in [lo, hi).
    #[inline]
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.next_f64() * (hi - lo)
    }
}

/// Build a random Coons patch that fully encloses the canvas. All 4 boundary
/// curves bulge OUTSIDE the canvas, so every pixel in [0,w) x [0,h) is inside
/// the patch — essential for using this as a background. Per-side amplitudes
/// and jittered tangent positions keep the shape asymmetric and interesting.
fn random_coons_patch(w: f64, h: f64, rng: &mut SplitMix64) -> CoonsPatch {
    // Canvas corners are the patch corners.
    let tl = Point::new(0.0, 0.0);
    let tr = Point::new(w, 0.0);
    let bl = Point::new(0.0, h);
    let br = Point::new(w, h);

    // How far each edge bows outward, as a fraction of the perpendicular dim.
    // Kept modest so the Newton solver in `inverse_eval` converges cleanly.
    let (lo, hi) = (0.08, 0.28);
    let top_amp = rng.range(lo, hi) * h;
    let bot_amp = rng.range(lo, hi) * h;
    let lef_amp = rng.range(lo, hi) * w;
    let rig_amp = rng.range(lo, hi) * w;

    // Along-axis positions and amplitude fractions for each control point.
    // Two CPs per curve; drawing them independently (rather than symmetrically)
    // lets curves become S-shaped, not just arcs.
    // Inlined instead of using a helper closure because two simultaneously-live
    // closures both capturing `&mut rng` would conflict under the borrow checker.
    let t_u1 = rng.range(0.15, 0.45);
    let t_u2 = rng.range(0.55, 0.85);
    let t_o1 = rng.range(0.5, 1.0);
    let t_o2 = rng.range(0.5, 1.0);

    let b_u1 = rng.range(0.15, 0.45);
    let b_u2 = rng.range(0.55, 0.85);
    let b_o1 = rng.range(0.5, 1.0);
    let b_o2 = rng.range(0.5, 1.0);

    let l_u1 = rng.range(0.15, 0.45);
    let l_u2 = rng.range(0.55, 0.85);
    let l_o1 = rng.range(0.5, 1.0);
    let l_o2 = rng.range(0.5, 1.0);

    let r_u1 = rng.range(0.15, 0.45);
    let r_u2 = rng.range(0.55, 0.85);
    let r_o1 = rng.range(0.5, 1.0);
    let r_o2 = rng.range(0.5, 1.0);

    CoonsPatch {
        // Top edge: TL -> TR, bowing up (negative y, above the canvas).
        top: CubicBez::new(
            tl,
            Point::new(t_u1 * w, -top_amp * t_o1),
            Point::new(t_u2 * w, -top_amp * t_o2),
            tr,
        ),
        // Bottom edge: BL -> BR, bowing down (y > h).
        bottom: CubicBez::new(
            bl,
            Point::new(b_u1 * w, h + bot_amp * b_o1),
            Point::new(b_u2 * w, h + bot_amp * b_o2),
            br,
        ),
        // Left edge: TL -> BL, bowing left (negative x).
        left: CubicBez::new(
            tl,
            Point::new(-lef_amp * l_o1, l_u1 * h),
            Point::new(-lef_amp * l_o2, l_u2 * h),
            bl,
        ),
        // Right edge: TR -> BR, bowing right (x > w).
        right: CubicBez::new(
            tr,
            Point::new(w + rig_amp * r_o1, r_u1 * h),
            Point::new(w + rig_amp * r_o2, r_u2 * h),
            br,
        ),
    }
}

/// Generate a harmonious 4-corner palette in Oklab. Picks a random base hue
/// and one of several classical color-theory schemes, jitters lightness and
/// chroma a bit, then shuffles which hue lands on which corner so the base
/// hue isn't always in the same place.
fn random_coons_colors(rng: &mut SplitMix64) -> PatchColors {
    use std::f64::consts::{PI, TAU};

    let base_hue = rng.range(0.0, TAU);

    // Hue offsets for a chosen harmony scheme.
    let scheme = rng.next_f64();
    let hue_offsets: [f64; 4] = if scheme < 0.40 {
        // Analogous -- 4 hues within ~30..70 deg of base. Calmest backgrounds.
        let span = rng.range(0.5, 1.2);
        [0.0, span / 3.0, 2.0 * span / 3.0, span]
    } else if scheme < 0.70 {
        // Split-complementary -- base, near-complement, complement, near-complement.
        let spread = rng.range(0.3, 0.6);
        [0.0, PI - spread, PI, PI + spread]
    } else if scheme < 0.90 {
        // Triadic plus one free accent hue.
        let third = TAU / 3.0;
        [0.0, third, 2.0 * third, rng.range(0.0, TAU)]
    } else {
        // Tetradic (square) -- most vibrant, used rarely.
        let q = TAU / 4.0;
        [0.0, q, 2.0 * q, 3.0 * q]
    };

    // Palette parameters. Moderate chroma keeps the background gentle enough
    // for overlaid text to stay readable.
    let base_l: f32 = rng.range(0.55, 0.82) as f32;
    let l_jitter: f32 = 0.12;
    let base_chroma: f32 = rng.range(0.05, 0.14) as f32;

    // Fisher-Yates shuffle so the base hue isn't pinned to top-left.
    let corner_perm: [usize; 4] = {
        let mut perm = [0usize, 1, 2, 3];
        for i in (1..4).rev() {
            let j = (rng.next_u64() % (i as u64 + 1)) as usize;
            perm.swap(i, j);
        }
        perm
    };

    // Build 4 Oklab colors with per-corner lightness and chroma jitter.
    // A plain loop keeps borrow rules trivial: `rng` is simply mutated
    // through its direct `&mut` reference without any closure captures.
    // `Oklab` is Copy (oklab crate derives it), so the `[x; 4]` shorthand
    // is fine for the placeholder array.
    let mut raw_colors: [(Oklab, f32); 4] = [(
        Oklab {
            l: 0.0,
            a: 0.0,
            b: 0.0,
        },
        255.0,
    ); 4];
    for i in 0..4 {
        let hue = base_hue + hue_offsets[i];
        let l = (base_l + (rng.range(-1.0, 1.0) as f32) * l_jitter).clamp(0.30, 0.95);
        let c = base_chroma * (rng.range(0.75, 1.25) as f32);
        let a = c * hue.cos() as f32;
        let b = c * hue.sin() as f32;
        raw_colors[i] = (Oklab { l, a, b }, 255.0);
    }

    PatchColors {
        top_left: raw_colors[corner_perm[0]],
        top_right: raw_colors[corner_perm[1]],
        bottom_left: raw_colors[corner_perm[2]],
        bottom_right: raw_colors[corner_perm[3]],
    }
}

/// Derive a seed from the OS entropy pool when the user didn't supply one.
/// Uses `getrandom::fill`; if the call fails (extremely rare on any supported
/// platform) we fall back to a fixed constant so the tool still produces an
/// output instead of aborting.
fn random_seed() -> u64 {
    let mut buf = [0u8; 8];
    if getrandom::fill(&mut buf).is_err() {
        return 0xDEAD_BEEF_CAFE_BABE;
    }
    u64::from_le_bytes(buf)
}

/// 1D cubic Bezier in value-space: `(y0, y1, y2, y3)` are the four ordinate
/// control values, evaluated at parameter `t` in `[0, 1]`. Returns roughly
/// `y0` at `t=0` and `y3` at `t=1`, with the intermediate shape pulled toward
/// `y1` and `y2`. Output is NOT clamped -- the caller decides what range
/// makes sense.
#[inline]
fn bezier1d(y0: f64, y1: f64, y2: f64, y3: f64, t: f64) -> f64 {
    let it = 1.0 - t;
    let it2 = it * it;
    let t2 = t * t;
    it2 * it * y0 + 3.0 * it2 * t * y1 + 3.0 * it * t2 * y2 + t2 * t * y3
}

/// Pre-computed 2D scalar field that drives the Coons canvas transparency.
///
/// Two random cubic Bezier "spine" curves are drawn across the canvas, each
/// sampled into a polyline. For each pixel we take the smaller of the two
/// distances to these polylines, normalize by half the canvas diagonal, then
/// reshape through a 1D cubic Bezier to get a scalar in `[0, 1]` that is
/// finally mapped to the user's `[t_min, t_max]` transparency range.
///
/// Close to a spine -> `t_min` (least transparent).
/// Far from both  -> `t_max` (most transparent).
struct TransparencyField {
    /// Sampled points along spine curve #1 (pixel coordinates).
    spine_a: Vec<Point>,
    /// Sampled points along spine curve #2.
    spine_b: Vec<Point>,
    /// `1 / (half canvas diagonal)`. Multiplied in to avoid a per-pixel divide.
    inv_max_dist: f64,
    /// User-chosen transparency bounds.
    t_min: f64,
    t_max: f64,
    /// Four ordinate control values for a 1D cubic Bezier that reshapes the
    /// normalized distance. `[y0, y1, y2, y3]`.
    shape: [f64; 4],
}

impl TransparencyField {
    /// Number of segments each spine bezier is sampled into. 32 is plenty
    /// for smooth distance estimates on canvases up to a few thousand px;
    /// loss of accuracy relative to analytic distance is well below one pixel.
    const SAMPLES: usize = 32;

    fn new(w: f64, h: f64, t_min: f64, t_max: f64, rng: &mut SplitMix64) -> Self {
        let spine_a = Self::sample_random_spine(w, h, rng);
        let spine_b = Self::sample_random_spine(w, h, rng);

        // Half the canvas diagonal is a pragmatic "max distance". Actual per-
        // pixel distances rarely approach this for well-placed spines, so the
        // reshape bezier does most of the useful range-spreading work.
        let max_dist = 0.5 * (w * w + h * h).sqrt();
        let inv_max_dist = if max_dist > 0.0 { 1.0 / max_dist } else { 0.0 };

        // Monotonically-increasing reshape curve: near the spine pixels stay
        // in low-transparency territory, far pixels swing up toward high.
        // Randomizing the two interior controls varies the steepness and
        // where the transition happens without flipping the direction.
        let shape = [
            rng.range(0.0, 0.15),
            rng.range(0.05, 0.55),
            rng.range(0.45, 0.95),
            rng.range(0.85, 1.0),
        ];

        Self {
            spine_a,
            spine_b,
            inv_max_dist,
            t_min,
            t_max,
            shape,
        }
    }

    /// Build one random 2D cubic Bezier spanning the canvas and return its
    /// polyline sampling. The curve enters on one canvas edge and exits on a
    /// different edge; the two interior control points are chosen freely in
    /// the canvas interior so the path can swoop and curl.
    fn sample_random_spine(w: f64, h: f64, rng: &mut SplitMix64) -> Vec<Point> {
        let edge_a = (rng.next_u64() % 4) as u32;
        let mut edge_b = (rng.next_u64() % 4) as u32;
        while edge_b == edge_a {
            edge_b = (rng.next_u64() % 4) as u32;
        }

        let p0 = edge_point(w, h, edge_a, rng.next_f64());
        let p3 = edge_point(w, h, edge_b, rng.next_f64());
        let p1 = Point::new(rng.range(0.1 * w, 0.9 * w), rng.range(0.1 * h, 0.9 * h));
        let p2 = Point::new(rng.range(0.1 * w, 0.9 * w), rng.range(0.1 * h, 0.9 * h));
        let curve = CubicBez::new(p0, p1, p2, p3);

        let n = Self::SAMPLES;
        let mut spine = Vec::with_capacity(n + 1);
        for i in 0..=n {
            let t = i as f64 / n as f64;
            spine.push(curve.eval(t));
        }
        spine
    }

    /// Transparency value (in `[t_min, t_max]`) at pixel `(x, y)`.
    #[inline]
    fn transparency_at(&self, x: f64, y: f64) -> f64 {
        // Min squared distance to either spine's polyline. Using d2 defers
        // the sqrt to a single call at the end.
        let mut min_d2 = f64::INFINITY;
        for sp in self.spine_a.iter().chain(self.spine_b.iter()) {
            let dx = x - sp.x;
            let dy = y - sp.y;
            let d2 = dx * dx + dy * dy;
            if d2 < min_d2 {
                min_d2 = d2;
            }
        }

        let d = min_d2.sqrt();
        let d_norm = (d * self.inv_max_dist).clamp(0.0, 1.0);

        let shaped = bezier1d(
            self.shape[0],
            self.shape[1],
            self.shape[2],
            self.shape[3],
            d_norm,
        )
        .clamp(0.0, 1.0);

        self.t_min + shaped * (self.t_max - self.t_min)
    }
}

/// Point on edge `edge` (0=top, 1=right, 2=bottom, 3=left) at parameter `t`
/// along that edge, in a w-by-h canvas. Used to pick bezier endpoints.
fn edge_point(w: f64, h: f64, edge: u32, t: f64) -> Point {
    match edge {
        0 => Point::new(t * w, 0.0), // top edge, left -> right
        1 => Point::new(w, t * h),   // right edge, top -> bottom
        2 => Point::new(t * w, h),   // bottom edge, left -> right
        _ => Point::new(0.0, t * h), // left edge, top -> bottom
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
        CanvasSpec::Linear {
            angle_deg,
            stops,
            easing,
        } => {
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
                        let mut t = if total_span > 0.0 {
                            ((proj + max_proj) / total_span).clamp(0.0, 1.0)
                        } else {
                            0.0
                        };

                        // Apply CSS easing if provided
                        if let Some(ease) = easing {
                            t = ease.ease(t);
                        }

                        let c = prepared.eval(t);
                        px.copy_from_slice(&c);
                    }
                });
        }
        CanvasSpec::Radial {
            center_x,
            center_y,
            stops,
            easing,
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
                        let mut t = (r * inv_max_r).clamp(0.0, 1.0);

                        // Apply CSS easing if provided
                        if let Some(ease) = easing {
                            t = ease.ease(t);
                        }

                        let c = prepared.eval(t);
                        px.copy_from_slice(&c);
                    }
                });
        }
        CanvasSpec::Mesh { colors } => {
            let width_f = (width as f64 - 1.0).max(1.0);
            let height_f = (height as f64 - 1.0).max(1.0);

            buf.par_chunks_mut(row_bytes)
                .enumerate()
                .for_each(|(y, row)| {
                    let v = y as f64 / height_f;
                    for (x, px) in row.chunks_exact_mut(4).enumerate() {
                        let u = x as f64 / width_f;
                        let c = colors.eval_color(u, v);
                        px.copy_from_slice(&c);
                    }
                });
        }
        CanvasSpec::Coons {
            seed,
            colors,
            easing,
            transparency,
        } => {
            let w = width as f64;
            let h = height as f64;

            // Resolve the seed: use the caller's or derive one from OS entropy.
            // When auto-seeding, report the chosen seed on stderr so the user
            // can capture an interesting output and reproduce it with `seed:N`.
            let actual_seed = match seed {
                Some(s) => *s,
                None => {
                    let s = random_seed();
                    eprintln!("canvas coons: using auto seed {}", s);
                    s
                }
            };
            let mut rng = SplitMix64::new(actual_seed);

            // Random geometry. Drawn from `rng` first so that seeds behave
            // consistently even when explicit colors are supplied.
            let patch = random_coons_patch(w, h, &mut rng);

            // Colors: either what the user passed, or a harmonious random palette
            // derived from the same PRNG stream.
            let patch_colors = match colors {
                Some(c) => c.clone(),
                None => random_coons_colors(&mut rng),
            };

            // Transparency field is built AFTER colors so that adding colors
            // doesn't shift the RNG stream feeding the transparency field --
            // and so omitting `transparency:` doesn't consume any RNG at all.
            let transparency_field: Option<TransparencyField> =
                if let Some((t_min, t_max)) = transparency {
                    Some(TransparencyField::new(w, h, *t_min, *t_max, &mut rng))
                } else {
                    None
                };

            // Inverse-evaluation fallback factors. When Newton-Raphson fails
            // for an edge pixel (rare with the modest bulges we generate),
            // we fall back to plain bilinear (x/w, y/h) coords so the pixel
            // still receives a sensible color instead of staying transparent.
            let inv_w = 1.0 / (w - 1.0).max(1.0);
            let inv_h = 1.0 / (h - 1.0).max(1.0);

            buf.par_chunks_mut(row_bytes)
                .enumerate()
                .for_each(|(y, row)| {
                    let v_fallback = y as f64 * inv_h;
                    let y_f = y as f64;
                    for (x, px) in row.chunks_exact_mut(4).enumerate() {
                        let pt = Point::new(x as f64, y_f);

                        // Solve for (u, v) via Newton-Raphson; fall back to
                        // bilinear if the solver bails, guaranteeing full
                        // coverage — critical when this canvas is a background.
                        let (mut u, mut v) = patch
                            .inverse_eval(pt)
                            .unwrap_or((x as f64 * inv_w, v_fallback));

                        // Apply CSS easing to u and v independently when set,
                        // yielding non-linear color transitions inside the patch.
                        if let Some(ease) = easing {
                            u = ease.ease(u);
                            v = ease.ease(v);
                        }

                        let mut c = patch_colors.eval_color(u, v);

                        // Modulate alpha by the transparency field if enabled.
                        // Multiplicative: existing alpha from color interp is
                        // preserved when the field returns 0.
                        if let Some(field) = &transparency_field {
                            let t = field.transparency_at(x as f64, y_f);
                            let alpha_factor = (1.0 - t) as f32;
                            c[3] = (c[3] as f32 * alpha_factor).round().clamp(0.0, 255.0) as u8;
                        }

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
            CanvasSpec::Linear {
                angle_deg, stops, ..
            } => {
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
                ..
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
                ..
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
                ..
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

    #[test]
    fn parse_arg_coons_minimal() {
        // Bare `coons` -- random shape AND random colors from an auto seed.
        let c = CanvasConfig::parse_arg("size:64x64,coons").unwrap();
        assert_eq!(c.size, Some((64, 64)));
        match c.spec {
            CanvasSpec::Coons {
                seed,
                colors,
                easing,
                transparency,
            } => {
                assert_eq!(seed, None);
                assert!(colors.is_none());
                assert!(easing.is_none());
                assert!(transparency.is_none());
            }
            _ => panic!("expected coons"),
        }
    }

    #[test]
    fn parse_arg_coons_transparency_range() {
        let c = CanvasConfig::parse_arg("size:64x64,coons,transparency:0.2-0.8").unwrap();
        match c.spec {
            CanvasSpec::Coons { transparency, .. } => {
                assert_eq!(transparency, Some((0.2, 0.8)));
            }
            _ => panic!("expected coons"),
        }
    }

    #[test]
    fn parse_arg_coons_transparency_single_value() {
        // Single-value form means MIN == MAX (uniform transparency field).
        let c = CanvasConfig::parse_arg("size:64x64,coons,transparency:0.5").unwrap();
        match c.spec {
            CanvasSpec::Coons { transparency, .. } => {
                assert_eq!(transparency, Some((0.5, 0.5)));
            }
            _ => panic!("expected coons"),
        }
    }

    #[test]
    fn parse_arg_coons_transparency_with_seed_and_colors() {
        let c = CanvasConfig::parse_arg(
            "size:64x64,coons,seed:7,transparency:0.0-1.0,#ff0000,#00ff00,#0000ff,#ffff00",
        )
        .unwrap();
        match c.spec {
            CanvasSpec::Coons {
                seed,
                colors,
                transparency,
                ..
            } => {
                assert_eq!(seed, Some(7));
                assert_eq!(transparency, Some((0.0, 1.0)));
                assert!(colors.is_some());
            }
            _ => panic!("expected coons"),
        }
    }

    #[test]
    fn parse_arg_coons_transparency_order_independent() {
        // `seed:`, `transparency:` and colors may appear in any order.
        let a = CanvasConfig::parse_arg("coons,transparency:0.1-0.9,seed:42").unwrap();
        let b = CanvasConfig::parse_arg("coons,seed:42,transparency:0.1-0.9").unwrap();
        assert_eq!(a.spec, b.spec);
    }

    #[test]
    fn parse_arg_coons_bad_transparency() {
        // Out-of-range rejected.
        assert!(CanvasConfig::parse_arg("coons,transparency:-0.1-0.5").is_err());
        assert!(CanvasConfig::parse_arg("coons,transparency:0.5-1.1").is_err());
        // min > max rejected.
        assert!(CanvasConfig::parse_arg("coons,transparency:0.8-0.2").is_err());
        // Non-numeric rejected.
        assert!(CanvasConfig::parse_arg("coons,transparency:abc").is_err());
        assert!(CanvasConfig::parse_arg("coons,transparency:0.2-xyz").is_err());
        // Missing side rejected.
        assert!(CanvasConfig::parse_arg("coons,transparency:0.2-").is_err());
        assert!(CanvasConfig::parse_arg("coons,transparency:-0.8").is_err());
        // Specified twice rejected.
        assert!(
            CanvasConfig::parse_arg("coons,transparency:0.2-0.8,transparency:0.3-0.7").is_err()
        );
    }

    #[test]
    fn parse_arg_coons_seed_only() {
        let c = CanvasConfig::parse_arg("coons,seed:42").unwrap();
        match c.spec {
            CanvasSpec::Coons { seed, colors, .. } => {
                assert_eq!(seed, Some(42));
                assert!(colors.is_none());
            }
            _ => panic!("expected coons"),
        }
    }

    #[test]
    fn parse_arg_coons_with_colors() {
        // Random shape, explicit colors -- preserves the old 4-color syntax.
        let c = CanvasConfig::parse_arg("coons,#ff0000,#00ff00,#0000ff,#ffff00").unwrap();
        match c.spec {
            CanvasSpec::Coons { seed, colors, .. } => {
                assert_eq!(seed, None);
                assert!(colors.is_some());
            }
            _ => panic!("expected coons"),
        }
    }

    #[test]
    fn parse_arg_coons_seed_and_colors() {
        let c = CanvasConfig::parse_arg("coons,seed:1234,#ff0000,#00ff00,#0000ff,#ffff00").unwrap();
        match c.spec {
            CanvasSpec::Coons { seed, colors, .. } => {
                assert_eq!(seed, Some(1234));
                assert!(colors.is_some());
            }
            _ => panic!("expected coons"),
        }
    }

    #[test]
    fn parse_arg_coons_bad_color_count() {
        // 1 or 3 colors -- neither "random palette" (0) nor "explicit" (4).
        assert!(CanvasConfig::parse_arg("coons,#ff0000").is_err());
        assert!(CanvasConfig::parse_arg("coons,#f00,#0f0,#00f").is_err());
    }

    #[test]
    fn parse_arg_coons_bad_seed() {
        // Non-numeric seed rejected.
        assert!(CanvasConfig::parse_arg("coons,seed:abc").is_err());
        // Seed specified twice rejected.
        assert!(CanvasConfig::parse_arg("coons,seed:42,seed:99").is_err());
    }

    #[test]
    fn splitmix64_deterministic() {
        // Identical seeds produce identical streams.
        let mut a = SplitMix64::new(0xDEAD_BEEF);
        let mut b = SplitMix64::new(0xDEAD_BEEF);
        for _ in 0..32 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn splitmix64_range_within_bounds() {
        let mut r = SplitMix64::new(0x1234_5678_9ABC_DEF0);
        for _ in 0..1000 {
            let v = r.range(-2.5, 7.25);
            assert!((-2.5..7.25).contains(&v));
            let f = r.next_f64();
            assert!((0.0..1.0).contains(&f));
        }
    }

    #[test]
    fn random_coons_colors_deterministic() {
        // Same seed -> same palette.
        let mut r1 = SplitMix64::new(7);
        let mut r2 = SplitMix64::new(7);
        let p1 = random_coons_colors(&mut r1);
        let p2 = random_coons_colors(&mut r2);
        assert_eq!(p1, p2);
    }

    #[test]
    fn random_coons_patch_encloses_canvas() {
        // Every bulge control point should sit outside the canvas rectangle,
        // which is what guarantees full pixel coverage for backgrounds.
        let mut rng = SplitMix64::new(0xABCD);
        let (w, h) = (100.0_f64, 80.0_f64);
        let patch = random_coons_patch(w, h, &mut rng);

        // Top curve: two interior control points must have y < 0.
        assert!(patch.top.p1.y < 0.0);
        assert!(patch.top.p2.y < 0.0);
        // Bottom curve: y > h.
        assert!(patch.bottom.p1.y > h);
        assert!(patch.bottom.p2.y > h);
        // Left curve: x < 0.
        assert!(patch.left.p1.x < 0.0);
        assert!(patch.left.p2.x < 0.0);
        // Right curve: x > w.
        assert!(patch.right.p1.x > w);
        assert!(patch.right.p2.x > w);

        // Corners must be exactly the canvas corners.
        assert_eq!(patch.top.p0, Point::new(0.0, 0.0));
        assert_eq!(patch.top.p3, Point::new(w, 0.0));
        assert_eq!(patch.bottom.p0, Point::new(0.0, h));
        assert_eq!(patch.bottom.p3, Point::new(w, h));
    }

    #[test]
    fn bezier1d_endpoints() {
        // t=0 -> y0, t=1 -> y3 (within floating point noise).
        assert!((bezier1d(0.2, 0.7, 0.3, 0.9, 0.0) - 0.2).abs() < 1e-12);
        assert!((bezier1d(0.2, 0.7, 0.3, 0.9, 1.0) - 0.9).abs() < 1e-12);
        // A straight line (controls on the line) evaluates linearly at t=0.5.
        assert!((bezier1d(0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0, 0.5) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn transparency_field_deterministic() {
        // Same seed -> identical field, identical samples.
        let (w, h, tmin, tmax) = (200.0, 150.0, 0.2, 0.8);
        let mut r1 = SplitMix64::new(0xF00D);
        let mut r2 = SplitMix64::new(0xF00D);
        let f1 = TransparencyField::new(w, h, tmin, tmax, &mut r1);
        let f2 = TransparencyField::new(w, h, tmin, tmax, &mut r2);
        for &(x, y) in &[(0.0, 0.0), (100.0, 75.0), (199.0, 149.0), (50.0, 120.0)] {
            assert_eq!(f1.transparency_at(x, y), f2.transparency_at(x, y));
        }
    }

    #[test]
    fn transparency_field_within_range() {
        // Sampled values must stay inside the requested [t_min, t_max] band.
        let (w, h, tmin, tmax) = (256.0, 256.0, 0.15, 0.85);
        let mut rng = SplitMix64::new(1);
        let field = TransparencyField::new(w, h, tmin, tmax, &mut rng);
        for y in (0..256).step_by(13) {
            for x in (0..256).step_by(13) {
                let t = field.transparency_at(x as f64, y as f64);
                assert!(
                    (tmin - 1e-9..=tmax + 1e-9).contains(&t),
                    "t={} outside [{}, {}] at ({}, {})",
                    t,
                    tmin,
                    tmax,
                    x,
                    y
                );
            }
        }
    }

    #[test]
    fn transparency_field_uniform_when_min_eq_max() {
        // With a degenerate range the field must return a constant, since
        // t_min + shaped * 0 == t_min for every pixel.
        let mut rng = SplitMix64::new(99);
        let field = TransparencyField::new(300.0, 200.0, 0.4, 0.4, &mut rng);
        let samples = [
            field.transparency_at(0.0, 0.0),
            field.transparency_at(150.0, 100.0),
            field.transparency_at(299.0, 199.0),
            field.transparency_at(42.0, 17.0),
        ];
        for s in samples {
            assert!((s - 0.4).abs() < 1e-12);
        }
    }
}
