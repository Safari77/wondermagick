use crate::{arg_parse_err::ArgParseErr, error::MagickError, image::Image};
use delaunator::{Point as DelaunayPoint, triangulate};
use image::{DynamicImage, ImageBuffer, Rgba};
use kurbo::{CubicBez, ParamCurve, Point};
use noise::{Fbm, MultiFractal, NoiseFn, Perlin, utils::PlaneMapBuilder};
use oklab::{
    Oklab, Rgb, linear_srgb_to_oklab, oklab_to_linear_srgb, oklab_to_srgb_f32, srgb_f32_to_oklab,
};
use rayon::prelude::*;

/// 16-bit RGBA image buffer. `image` keeps its own `Rgba16Image` alias
/// crate-private, so we spell the type out once here.
type Rgba16Image = ImageBuffer<Rgba<u16>, Vec<u16>>;

/// Full-scale value of a 16-bit channel, as a float. Used for every
/// normalize/denormalize step so the magic number lives in exactly one place.
const MAX16: f32 = u16::MAX as f32;

/// Expands an 8-bit channel to 16-bit. Multiplying by 257 (rather than
/// shifting left by 8) maps 0x00 -> 0x0000 and 0xFF -> 0xFFFF exactly,
/// so pure white stays pure white.
#[inline]
fn expand8(v: u8) -> u16 {
    v as u16 * 257
}

/// Expands a 4-bit channel (a single hex digit) to 16-bit: 0xF -> 0xFFFF.
#[inline]
fn expand4(v: u8) -> u16 {
    v as u16 * 0x1111
}

/// Quantizes one gamma-encoded sRGB channel in [0.0, 1.0] to 16 bits.
#[inline]
fn quantize16(v: f32) -> u16 {
    // `oklab_to_srgb_f32` already clamps its own output, but out-of-gamut
    // values can still reach us from interpolation, so clamp again. The
    // float->int `as` cast saturates and maps NaN to 0 rather than wrapping.
    (v.clamp(0.0, 1.0) * MAX16 + 0.5) as u16
}

/// Converts an Oklab color to gamma-encoded 16-bit sRGB.
/// Goes through the crate's f32 conversion rather than `oklab_to_srgb`,
/// which would quantize to 8 bits and throw away the precision we want.
#[inline]
fn oklab_to_rgb16(c: Oklab) -> [u16; 3] {
    let rgb = oklab_to_srgb_f32(c);
    [quantize16(rgb.r), quantize16(rgb.g), quantize16(rgb.b)]
}

/// Gamma-encodes a linear-light sRGB triple to 16-bit sRGB. Round-tripping
/// through Oklab uses the crate's public conversions in both directions
/// instead of hand-rolling the sRGB transfer function; the detour costs about
/// one code point of accuracy at 16 bits.
#[inline]
fn linear_rgb_to_rgb16(r: f32, g: f32, b: f32) -> [u16; 3] {
    oklab_to_rgb16(linear_srgb_to_oklab(Rgb { r, g, b }))
}

/// Converts a straight-alpha 16-bit sRGB color to `(Oklab, alpha)`.
/// The alpha channel is passed through untouched -- only RGB is color-managed.
#[inline]
fn rgba16_to_oklab(c: [u16; 4]) -> (Oklab, u16) {
    let lab = srgb_f32_to_oklab(Rgb {
        r: c[0] as f32 / MAX16,
        g: c[1] as f32 / MAX16,
        b: c[2] as f32 / MAX16,
    });
    (lab, c[3])
}

/// One stop in a gradient.
#[derive(Debug, Clone, PartialEq)]
pub struct GradientStop {
    /// Position in [0.0, 1.0]
    pub pos: f64,
    /// Straight (non-premultiplied) 16-bit RGBA
    pub color: [u16; 4],
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
        Self { p1: (x1, y1), p2: (x2, y2) }
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

        // 20 iterations provide ~1e-6 precision in `t`. 12 (~0.00024) was fine
        // for 8-bit output but is coarse enough to show up as stair-stepping in
        // a 16-bit ramp, where one code point is only 1.5e-5 of the range.
        for _ in 0..20 {
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
/// The `f32` beside each `Oklab` is straight alpha on the 16-bit scale
/// (0.0 = transparent, 65535.0 = opaque).
#[derive(Debug, Clone, PartialEq)]
pub struct MeshColors {
    pub tl: (Oklab, f32),
    pub tr: (Oklab, f32),
    pub bl: (Oklab, f32),
    pub br: (Oklab, f32),
}

impl MeshColors {
    #[inline]
    pub fn eval_color(&self, u: f64, v: f64) -> [u16; 4] {
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

        let rgb = oklab_to_rgb16(mixed_oklab);

        // Return the RGB along with the calculated alpha
        [rgb[0], rgb[1], rgb[2], mixed_alpha.round().clamp(0.0, MAX16) as u16]
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
            if !(-0.2..=1.2).contains(&u) || !(-0.2..=1.2).contains(&v) {
                return None;
            }
        }

        // Final fallback check
        if (0.0..=1.0).contains(&u) && (0.0..=1.0).contains(&v) { Some((u, v)) } else { None }
    }
}

/// Interpolates 4 corner colors using Bilinear interpolation in Oklab space, preserving Alpha.
/// The `f32` beside each `Oklab` is straight alpha on the 16-bit scale
/// (0.0 = transparent, 65535.0 = opaque).
#[derive(Debug, Clone, PartialEq)]
pub struct PatchColors {
    pub top_left: (Oklab, f32),
    pub top_right: (Oklab, f32),
    pub bottom_left: (Oklab, f32),
    pub bottom_right: (Oklab, f32),
}

impl PatchColors {
    #[inline]
    pub fn eval_color(&self, u: f64, v: f64) -> [u16; 4] {
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

        let rgb = oklab_to_rgb16(mixed_oklab);

        // Return the RGB along with the calculated alpha
        [rgb[0], rgb[1], rgb[2], mixed_alpha.round().clamp(0.0, MAX16) as u16]
    }
}

/// A coordinate for gradient centers, either a ratio [0.0, 1.0] or absolute pixels.
#[derive(Debug, Clone, PartialEq)]
pub enum Coord {
    Ratio(f64),
    Pixels(f64),
}

/// Edge treatment for the `voronoi` canvas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoronoiStyle {
    /// Nearest-seed lookup: each cell is a flat polygon with hard edges.
    Sharp,
    /// Distance-weighted blend of the surrounding seeds, which rounds the cell
    /// boundaries off into organic, metaball-like blobs.
    Blob,
}

/// What to fill the canvas with.
#[derive(Debug, Clone, PartialEq)]
pub enum CanvasSpec {
    /// Solid 16-bit RGBA fill
    Solid([u16; 4]),
    /// Linear gradient at an arbitrary angle (in degrees).
    /// 0 deg = left-to-right, 90 deg = top-to-bottom (angle increases clockwise
    /// in screen coordinates).
    Linear { angle_deg: f64, stops: Vec<GradientStop>, easing: Option<CssEasing> },
    /// Radial gradient. `center_x` and `center_y` define the center. 0 at center, 1 at the farthest corner.
    Radial { center_x: Coord, center_y: Coord, stops: Vec<GradientStop>, easing: Option<CssEasing> },
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
    /// Voronoi cells over a jittered grid, either flat-shaded with hard edges
    /// or blended into blobs.
    Voronoi {
        seed: Option<u64>,
        /// Number of cells across the short edge of the canvas.
        cells: u32,
        style: VoronoiStyle,
        /// Blob mode only: blend width between neighbouring cells, in cell
        /// widths. Small values approach `sharp`, large values smear together.
        softness: f64,
        /// Explicit cell palette. If `None`, a harmonious one is generated.
        colors: Option<Vec<[u16; 4]>>,
    },
    /// Fractal Brownian motion -- summed Perlin octaves from the `noise`
    /// crate -- mapped through a color ramp.
    Fbm {
        seed: Option<u64>,
        octaves: usize,
        frequency: f64,
        lacunarity: f64,
        persistence: f64,
        /// Half-extent of the sampled noise domain across the short edge.
        /// Larger values zoom out and pack in more detail.
        zoom: f64,
        /// Make the field tile seamlessly across the canvas edges.
        seamless: bool,
        /// Ramp the noise value is mapped through. If `None`, one is generated
        /// from the seed.
        stops: Option<Vec<GradientStop>>,
        easing: Option<CssEasing>,
    },
    /// Hair-like streamlines traced through a Perlin flow field.
    Flow {
        seed: Option<u64>,
        /// Approximate strand count; the real count is rounded to a grid.
        strands: u32,
        /// Integration steps per strand, i.e. how long a strand can grow.
        steps: u32,
        /// Length of one integration step, in pixels.
        step_len: f64,
        /// Half-extent of the noise domain; larger means finer, curlier flow.
        zoom: f64,
        /// How many half-turns the field angle spans.
        turns: f64,
        /// Strand thickness in pixels.
        width: f64,
        /// Per-step strand opacity in [0, 1].
        alpha: f64,
        colors: Option<Vec<[u16; 4]>>,
    },
    /// Low-poly facets: a Delaunay triangulation of jittered points, each
    /// triangle filled from an underlying color field.
    LowPoly {
        seed: Option<u64>,
        /// Approximate number of interior points.
        points: u32,
        /// Interpolate across each triangle instead of flat-filling it.
        smooth: bool,
        colors: Option<Vec<[u16; 4]>>,
    },
    /// Fractal flame: an iterated function system rendered through a
    /// log-density histogram.
    Flame {
        seed: Option<u64>,
        /// Chaos-game iterations per pixel. Higher is smoother and slower.
        quality: u32,
        /// Number of affine transforms in the system.
        transforms: u32,
        /// Tone-mapping gamma. Higher lifts the faint density regions.
        gamma: f64,
        /// Restrict the variation pool to the cut-free set, so the image has
        /// no hard seams. Off by default, which keeps the full set -- and the
        /// output of every seed rendered before this option existed.
        continuous: bool,
        colors: Option<Vec<[u16; 4]>>,
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
    ///   [size:WxH,]voronoi[,sharp|blob][,cells:N][,softness:F][,seed:N][,COLOR...]
    ///   [size:WxH,]fbm[,seed:N][,octaves:N][,freq:F][,lacunarity:F][,persistence:F]
    ///                 [,zoom:F][,seamless][,STOP...]
    ///   [size:WxH,]flow[,seed:N][,strands:N][,steps:N][,step:F][,zoom:F][,turns:F]
    ///                  [,width:F][,alpha:F][,COLOR...]
    ///   [size:WxH,]lowpoly[,seed:N][,points:N][,smooth][,COLOR...]
    ///   [size:WxH,]flame[,seed:N][,quality:N][,transforms:N][,gamma:F][,continuous][,COLOR...]
    ///
    /// When `size:WxH` is omitted, the canvas operation overwrites the current
    /// image's pixels while keeping its dimensions. When `size:WxH` is given,
    /// the current image is replaced with a new one of the requested size --
    /// useful when canvas is the first operation in the pipeline and no input
    /// image was loaded from disk.
    ///
    /// COLOR: hex, one of `#RGB`, `#RGBA`, `#RRGGBB`, `#RRGGBBAA` (8-bit forms,
    ///        expanded to the full 16-bit range), or `#RRRRGGGGBBBB` /
    ///        `#RRRRGGGGBBBBAAAA` for native 16-bit channels.
    ///
    /// STOP: either `POS:COLOR` (e.g. `0.5:#ff0000`), or just `COLOR` in which
    ///       case positions are evenly distributed in [0, 1]. Positional and
    ///       non-positional stops cannot be mixed in one spec.
    ///
    /// ANGLE_DEG: fractional degrees, e.g. `16.514`.
    ///
    /// Every generator that takes a `seed:` reports an auto-chosen seed on
    /// stderr, so an interesting result can be pinned down and reproduced.
    /// Their options may appear in any order, and any color list they accept
    /// replaces the palette that would otherwise be generated from the seed.
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
                "canvas: missing type (expected 'solid', 'linear', 'radial', 'mesh', \
             'coons', 'voronoi', 'fbm', 'flow', 'lowpoly', or 'flame')",
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
                if let Ok(p) = pts
                    && p.len() == 4
                {
                    easing = Some(CssEasing::new(p[0], p[1], p[2], p[3]));
                    continue;
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
                CanvasSpec::Linear { angle_deg, stops, easing }
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
                CanvasSpec::Radial { center_x, center_y, stops, easing }
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

                let to_oklab = |c: [u16; 4]| {
                    let (lab, a) = rgba16_to_oklab(c);
                    (lab, a as f32)
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

                        let to_oklab = |c: [u16; 4]| {
                            let (lab, a) = rgba16_to_oklab(c);
                            (lab, a as f32)
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

                CanvasSpec::Coons { seed, colors, easing, transparency }
            }
            "voronoi" => {
                let mut tok = SpecTokens::split(&remaining[1..]);
                let style = if tok.flag("blob") {
                    VoronoiStyle::Blob
                } else {
                    // `sharp` is the default; accepting it explicitly keeps
                    // the two styles symmetrical in the syntax.
                    tok.flag("sharp");
                    VoronoiStyle::Sharp
                };
                let seed = opt_seed(tok.take("seed"))?;
                let cells = opt_u32(
                    tok.take("cells"),
                    12,
                    1,
                    4096,
                    "canvas voronoi: invalid 'cells' (expected an integer in [1, 4096])",
                )?;
                let softness = opt_f64(
                    tok.take("softness"),
                    0.25,
                    0.001,
                    4.0,
                    "canvas voronoi: invalid 'softness' (expected a number in [0.001, 4.0])",
                )?;
                let colors = parse_color_list(
                    &tok.colors,
                    "canvas voronoi: needs at least 2 colors when colors are given",
                )?;
                tok.finish(
                    "canvas voronoi: unknown or repeated option (expected \
                     'sharp', 'blob', 'cells:N', 'softness:F', 'seed:N')",
                )?;
                CanvasSpec::Voronoi { seed, cells, style, softness, colors }
            }
            "fbm" => {
                let mut tok = SpecTokens::split(&remaining[1..]);
                let seamless = tok.flag("seamless");
                let seed = opt_seed(tok.take("seed"))?;
                let octaves = opt_u32(
                    tok.take("octaves"),
                    6,
                    1,
                    32,
                    "canvas fbm: invalid 'octaves' (expected an integer in [1, 32])",
                )? as usize;
                let frequency = opt_f64(
                    tok.take("freq"),
                    1.0,
                    1e-6,
                    1e6,
                    "canvas fbm: invalid 'freq' (expected a positive number)",
                )?;
                let lacunarity = opt_f64(
                    tok.take("lacunarity"),
                    std::f64::consts::PI * 2.0 / 3.0,
                    1.0,
                    8.0,
                    "canvas fbm: invalid 'lacunarity' (expected a number in [1.0, 8.0])",
                )?;
                let persistence = opt_f64(
                    tok.take("persistence"),
                    0.5,
                    0.01,
                    1.0,
                    "canvas fbm: invalid 'persistence' (expected a number in [0.01, 1.0])",
                )?;
                let zoom = opt_f64(
                    tok.take("zoom"),
                    3.0,
                    1e-3,
                    1e4,
                    "canvas fbm: invalid 'zoom' (expected a positive number)",
                )?;
                // The tail is a gradient, so it goes through the same stop
                // parser the linear and radial gradients use.
                let stops =
                    if tok.colors.is_empty() { None } else { Some(parse_stops(&tok.colors)?) };
                tok.finish(
                    "canvas fbm: unknown or repeated option (expected 'seed:N', \
                     'octaves:N', 'freq:F', 'lacunarity:F', 'persistence:F', \
                     'zoom:F', 'seamless')",
                )?;
                CanvasSpec::Fbm {
                    seed,
                    octaves,
                    frequency,
                    lacunarity,
                    persistence,
                    zoom,
                    seamless,
                    stops,
                    easing,
                }
            }
            "flow" => {
                let mut tok = SpecTokens::split(&remaining[1..]);
                let seed = opt_seed(tok.take("seed"))?;
                let strands = opt_u32(
                    tok.take("strands"),
                    2000,
                    1,
                    1_000_000,
                    "canvas flow: invalid 'strands' (expected an integer in [1, 1000000])",
                )?;
                let steps = opt_u32(
                    tok.take("steps"),
                    300,
                    1,
                    100_000,
                    "canvas flow: invalid 'steps' (expected an integer in [1, 100000])",
                )?;
                let step_len = opt_f64(
                    tok.take("step"),
                    1.5,
                    0.05,
                    64.0,
                    "canvas flow: invalid 'step' (expected a number in [0.05, 64.0])",
                )?;
                let zoom = opt_f64(
                    tok.take("zoom"),
                    2.0,
                    1e-3,
                    1e4,
                    "canvas flow: invalid 'zoom' (expected a positive number)",
                )?;
                let turns = opt_f64(
                    tok.take("turns"),
                    2.0,
                    0.0,
                    64.0,
                    "canvas flow: invalid 'turns' (expected a number in [0.0, 64.0])",
                )?;
                let width = opt_f64(
                    tok.take("width"),
                    1.0,
                    0.5,
                    64.0,
                    "canvas flow: invalid 'width' (expected a number in [0.5, 64.0])",
                )?;
                let alpha = opt_f64(
                    tok.take("alpha"),
                    0.35,
                    0.0,
                    1.0,
                    "canvas flow: invalid 'alpha' (expected a number in [0.0, 1.0])",
                )?;
                let colors = parse_color_list(
                    &tok.colors,
                    "canvas flow: needs at least 2 colors when colors are given",
                )?;
                tok.finish(
                    "canvas flow: unknown or repeated option (expected 'seed:N', \
                     'strands:N', 'steps:N', 'step:F', 'zoom:F', 'turns:F', \
                     'width:F', 'alpha:F')",
                )?;
                CanvasSpec::Flow {
                    seed,
                    strands,
                    steps,
                    step_len,
                    zoom,
                    turns,
                    width,
                    alpha,
                    colors,
                }
            }
            "lowpoly" => {
                let mut tok = SpecTokens::split(&remaining[1..]);
                let smooth = tok.flag("smooth");
                let seed = opt_seed(tok.take("seed"))?;
                let points = opt_u32(
                    tok.take("points"),
                    150,
                    3,
                    200_000,
                    "canvas lowpoly: invalid 'points' (expected an integer in [3, 200000])",
                )?;
                let colors = parse_color_list(
                    &tok.colors,
                    "canvas lowpoly: needs at least 2 colors when colors are given",
                )?;
                tok.finish(
                    "canvas lowpoly: unknown or repeated option (expected 'seed:N', \
                     'points:N', 'smooth')",
                )?;
                CanvasSpec::LowPoly { seed, points, smooth, colors }
            }
            "flame" => {
                let mut tok = SpecTokens::split(&remaining[1..]);
                let continuous = tok.flag("continuous");
                let seed = opt_seed(tok.take("seed"))?;
                let quality = opt_u32(
                    tok.take("quality"),
                    12,
                    1,
                    4096,
                    "canvas flame: invalid 'quality' (expected an integer in [1, 4096])",
                )?;
                let transforms = opt_u32(
                    tok.take("transforms"),
                    3,
                    2,
                    12,
                    "canvas flame: invalid 'transforms' (expected an integer in [2, 12])",
                )?;
                let gamma = opt_f64(
                    tok.take("gamma"),
                    2.2,
                    0.1,
                    10.0,
                    "canvas flame: invalid 'gamma' (expected a number in [0.1, 10.0])",
                )?;
                let colors = parse_color_list(
                    &tok.colors,
                    "canvas flame: needs at least 2 colors when colors are given",
                )?;
                tok.finish(
                    "canvas flame: unknown or repeated option (expected 'seed:N', \
                     'quality:N', 'transforms:N', 'gamma:F', 'continuous')",
                )?;
                CanvasSpec::Flame { seed, quality, transforms, gamma, continuous, colors }
            }
            _ => {
                return Err(ArgParseErr::with_msg(
                    "canvas: type must be 'solid', 'linear', 'radial', 'mesh', 'coons', \
                     'voronoi', 'fbm', 'flow', 'lowpoly', or 'flame'",
                ));
            }
        };

        Ok(Self { size, spec })
    }
}

fn parse_size(s: &str) -> Result<(u32, u32), ArgParseErr> {
    // accept both 'x' and 'X' as separator
    let mut it = s.splitn(2, ['x', 'X']);
    let w_str = it.next().ok_or_else(|| ArgParseErr::with_msg("canvas size: missing width"))?;
    let h_str = it.next().ok_or_else(|| {
        ArgParseErr::with_msg("canvas size: missing height (expected WIDTHxHEIGHT)")
    })?;
    let w: u32 = w_str.parse().map_err(|_| ArgParseErr::with_msg("canvas size: invalid width"))?;
    let h: u32 = h_str.parse().map_err(|_| ArgParseErr::with_msg("canvas size: invalid height"))?;
    if w == 0 || h == 0 {
        return Err(ArgParseErr::with_msg("canvas size: width and height must be greater than 0"));
    }
    // Guard against astronomically large allocations that would overflow usize.
    // 4 channels at 2 bytes each, since the canvas buffer is 16 bits per channel.
    let bytes = (w as u64)
        .checked_mul(h as u64)
        .and_then(|p| p.checked_mul(4 * std::mem::size_of::<u16>() as u64));
    if bytes.is_none_or(|b| b > (isize::MAX as u64)) {
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

/// Four hex digits -> one native 16-bit channel value.
fn hex_u16(d: &[u8]) -> Result<u16, ArgParseErr> {
    Ok(((hex_digit(d[0])? as u16) << 12)
        | ((hex_digit(d[1])? as u16) << 8)
        | ((hex_digit(d[2])? as u16) << 4)
        | (hex_digit(d[3])? as u16))
}

fn parse_coord(s: &str) -> Option<Coord> {
    if let Some(px_str) = s.strip_suffix("px") {
        let px: f64 = px_str.parse().ok()?;
        if px >= 0.0 {
            return Some(Coord::Pixels(px));
        }
    } else if let Ok(ratio) = s.parse::<f64>()
        && (0.0..=1.0).contains(&ratio)
    {
        return Some(Coord::Ratio(ratio));
    }
    None
}

/// Parses a hex color into straight 16-bit RGBA. The short (4-bit) and
/// byte-sized (8-bit) forms are expanded to the full 16-bit range, so
/// `#fff`, `#ffffff` and `#ffffffffffff` all mean the same pure white.
fn parse_color(s: &str) -> Result<[u16; 4], ArgParseErr> {
    let s = s.trim();
    let hex = s.strip_prefix('#').ok_or_else(|| {
        ArgParseErr::with_msg(
            "canvas color: must be hex like #RGB, #RGBA, #RRGGBB, #RRGGBBAA, \
             #RRRRGGGGBBBB, or #RRRRGGGGBBBBAAAA",
        )
    })?;
    let b = hex.as_bytes();
    match b.len() {
        3 => {
            // #RGB
            let r = hex_digit(b[0])?;
            let g = hex_digit(b[1])?;
            let bl = hex_digit(b[2])?;
            Ok([expand4(r), expand4(g), expand4(bl), u16::MAX])
        }
        4 => {
            // #RGBA
            let r = hex_digit(b[0])?;
            let g = hex_digit(b[1])?;
            let bl = hex_digit(b[2])?;
            let a = hex_digit(b[3])?;
            Ok([expand4(r), expand4(g), expand4(bl), expand4(a)])
        }
        6 => {
            // #RRGGBB
            Ok([
                expand8(hex_byte(b[0], b[1])?),
                expand8(hex_byte(b[2], b[3])?),
                expand8(hex_byte(b[4], b[5])?),
                u16::MAX,
            ])
        }
        8 => {
            // #RRGGBBAA
            Ok([
                expand8(hex_byte(b[0], b[1])?),
                expand8(hex_byte(b[2], b[3])?),
                expand8(hex_byte(b[4], b[5])?),
                expand8(hex_byte(b[6], b[7])?),
            ])
        }
        12 => {
            // #RRRRGGGGBBBB -- native 16-bit channels
            Ok([hex_u16(&b[0..4])?, hex_u16(&b[4..8])?, hex_u16(&b[8..12])?, u16::MAX])
        }
        16 => {
            // #RRRRGGGGBBBBAAAA -- native 16-bit channels with alpha
            Ok([hex_u16(&b[0..4])?, hex_u16(&b[4..8])?, hex_u16(&b[8..12])?, hex_u16(&b[12..16])?])
        }
        _ => Err(ArgParseErr::with_msg(
            "canvas color: expected #RGB, #RGBA, #RRGGBB, #RRGGBBAA, \
             #RRRRGGGGBBBB, or #RRRRGGGGBBBBAAAA",
        )),
    }
}

fn parse_stops(tokens: &[&str]) -> Result<Vec<GradientStop>, ArgParseErr> {
    if tokens.len() < 2 {
        return Err(ArgParseErr::with_msg("canvas gradient: at least 2 stops required"));
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
    stops.sort_by(|a, b| a.pos.partial_cmp(&b.pos).unwrap_or(std::cmp::Ordering::Equal));
    Ok(stops)
}

/// Type-specific tokens split into `key:value` options, bare flag words, and
/// colors. Every generator accepts its options in any order, so they all share
/// this instead of hand-rolling a scan each.
struct SpecTokens<'a> {
    opts: Vec<(&'a str, &'a str)>,
    flags: Vec<&'a str>,
    colors: Vec<&'a str>,
}

impl<'a> SpecTokens<'a> {
    fn split(tokens: &[&'a str]) -> Self {
        let mut opts = Vec::new();
        let mut flags = Vec::new();
        let mut colors = Vec::new();
        for &t in tokens {
            if t.starts_with('#') {
                colors.push(t);
            } else if let Some((k, v)) = t.split_once(':') {
                if v.trim_start().starts_with('#') {
                    // A positional gradient stop like `0.5:#ff0000`, not an
                    // option -- hand it to `parse_stops` intact.
                    colors.push(t);
                } else {
                    opts.push((k.trim(), v.trim()));
                }
            } else if !t.is_empty() {
                flags.push(t);
            }
        }
        Self { opts, flags, colors }
    }

    /// Removes the first `key:value` with this key and returns its value.
    /// A repeat is deliberately left behind so `finish` rejects it, which
    /// means duplicates and typos are caught by the same check.
    fn take(&mut self, key: &str) -> Option<&'a str> {
        let mut found = None;
        self.opts.retain(|(k, v)| {
            if found.is_none() && *k == key {
                found = Some(*v);
                false
            } else {
                true
            }
        });
        found
    }

    /// Removes a bare flag word, reporting whether it was present.
    fn flag(&mut self, name: &str) -> bool {
        let before = self.flags.len();
        self.flags.retain(|f| !f.eq_ignore_ascii_case(name));
        self.flags.len() != before
    }

    /// Errors if anything was left unconsumed.
    fn finish(&self, err: &'static str) -> Result<(), ArgParseErr> {
        if self.opts.is_empty() && self.flags.is_empty() {
            Ok(())
        } else {
            Err(ArgParseErr::with_msg(err))
        }
    }
}

/// Parses an optional integer option, range-checked, falling back to `default`.
fn opt_u32(
    v: Option<&str>,
    default: u32,
    min: u32,
    max: u32,
    err: &'static str,
) -> Result<u32, ArgParseErr> {
    match v {
        None => Ok(default),
        Some(s) => {
            let n: u32 = s.trim().parse().map_err(|_| ArgParseErr::with_msg(err))?;
            if n < min || n > max {
                return Err(ArgParseErr::with_msg(err));
            }
            Ok(n)
        }
    }
}

/// Parses an optional float option, range-checked, falling back to `default`.
fn opt_f64(
    v: Option<&str>,
    default: f64,
    min: f64,
    max: f64,
    err: &'static str,
) -> Result<f64, ArgParseErr> {
    match v {
        None => Ok(default),
        Some(s) => {
            let x: f64 = s.trim().parse().map_err(|_| ArgParseErr::with_msg(err))?;
            if !x.is_finite() || x < min || x > max {
                return Err(ArgParseErr::with_msg(err));
            }
            Ok(x)
        }
    }
}

/// Parses the shared `seed:N` option.
fn opt_seed(v: Option<&str>) -> Result<Option<u64>, ArgParseErr> {
    match v {
        None => Ok(None),
        Some(s) => s.trim().parse::<u64>().map(Some).map_err(|_| {
            ArgParseErr::with_msg(
                "canvas: invalid seed (expected non-negative integer fitting in u64)",
            )
        }),
    }
}

/// Parses a generator's optional palette. An empty list means "generate one
/// from the seed"; a list of one color is rejected, since every generator that
/// takes a palette needs at least two entries to interpolate between.
fn parse_color_list(
    tokens: &[&str],
    err: &'static str,
) -> Result<Option<Vec<[u16; 4]>>, ArgParseErr> {
    if tokens.is_empty() {
        return Ok(None);
    }
    if tokens.len() < 2 {
        return Err(ArgParseErr::with_msg(err));
    }
    let mut out = Vec::with_capacity(tokens.len());
    for t in tokens {
        out.push(parse_color(t)?);
    }
    Ok(Some(out))
}

/// A gradient pre-converted to Oklab for fast per-pixel evaluation.
/// Interpolating in Oklab gives perceptually uniform, visually pleasing
/// transitions (no muddy midpoints between complementary hues).
struct PreparedGradient {
    /// (position, oklab color, straight 16-bit alpha)
    stops: Vec<(f64, Oklab, u16)>,
}

impl PreparedGradient {
    fn new(stops: &[GradientStop]) -> Self {
        let prepared = stops
            .iter()
            .map(|s| {
                let (lab, alpha) = rgba16_to_oklab(s.color);
                (s.pos, lab, alpha)
            })
            .collect();
        Self { stops: prepared }
    }

    /// Builds a ramp directly from Oklab stops, skipping the sRGB round-trip
    /// `new` performs. Stops must already be sorted by position.
    fn from_oklab(stops: Vec<(f64, Oklab, u16)>) -> Self {
        Self { stops }
    }

    #[inline]
    fn eval(&self, t: f64) -> [u16; 4] {
        if self.stops.is_empty() {
            return [0, 0, 0, u16::MAX];
        }
        // Clamp-to-edge behaviour (t outside [first_pos, last_pos] -> edge color).
        let first = &self.stops[0];
        if self.stops.len() == 1 || t <= first.0 {
            let rgb = oklab_to_rgb16(first.1);
            return [rgb[0], rgb[1], rgb[2], first.2];
        }
        let last = self.stops.last().unwrap();
        if t >= last.0 {
            let rgb = oklab_to_rgb16(last.1);
            return [rgb[0], rgb[1], rgb[2], last.2];
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
        let rgb = oklab_to_rgb16(mixed);
        // Alpha is interpolated linearly in straight-alpha space.
        let alpha = (a_lo as f64 + (a_hi as f64 - a_lo as f64) * local_t)
            .round()
            .clamp(0.0, u16::MAX as f64) as u16;
        [rgb[0], rgb[1], rgb[2], alpha]
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
    let mut raw_colors: [(Oklab, f32); 4] = [(Oklab { l: 0.0, a: 0.0, b: 0.0 }, MAX16); 4];
    for i in 0..4 {
        let hue = base_hue + hue_offsets[i];
        let l = (base_l + (rng.range(-1.0, 1.0) as f32) * l_jitter).clamp(0.30, 0.95);
        let c = base_chroma * (rng.range(0.75, 1.25) as f32);
        let a = c * hue.cos() as f32;
        let b = c * hue.sin() as f32;
        raw_colors[i] = (Oklab { l, a, b }, MAX16);
    }

    PatchColors {
        top_left: raw_colors[corner_perm[0]],
        top_right: raw_colors[corner_perm[1]],
        bottom_left: raw_colors[corner_perm[2]],
        bottom_right: raw_colors[corner_perm[3]],
    }
}

/// A harmonious palette of `n` Oklab colors, alpha at full 16-bit scale.
///
/// This is the same hue-harmony idea as `random_coons_colors`, generalized to
/// any count. It is kept as a separate function rather than folding that one
/// into it, because changing the order `random_coons_colors` draws from the
/// PRNG would change the image every existing `coons,seed:N` produces.
fn random_palette(n: usize, rng: &mut SplitMix64) -> Vec<(Oklab, f32)> {
    use std::f64::consts::{PI, TAU};

    let n = n.max(1);
    let base_hue = rng.range(0.0, TAU);

    // Total hue sweep across the palette, picked from the same classical
    // schemes the coons palette uses.
    let scheme = rng.next_f64();
    let span = if scheme < 0.40 {
        rng.range(0.5, 1.2) // analogous -- calmest
    } else if scheme < 0.70 {
        PI + rng.range(-0.6, 0.6) // split-complementary
    } else if scheme < 0.90 {
        TAU * 2.0 / 3.0 // triadic
    } else {
        TAU // full wheel, most vibrant
    };

    let base_l = rng.range(0.42, 0.78) as f32;
    let l_span = rng.range(0.10, 0.36) as f32;
    let base_chroma = rng.range(0.05, 0.15) as f32;

    (0..n)
        .map(|i| {
            let t = if n > 1 { i as f64 / (n - 1) as f64 } else { 0.5 };
            let hue = base_hue + span * t;
            // Ramp lightness across the palette so generators that read it as
            // a gradient get contrast, not just a hue shift.
            let l =
                (base_l + l_span * (t as f32 - 0.5) * 2.0 + (rng.range(-1.0, 1.0) as f32) * 0.04)
                    .clamp(0.20, 0.96);
            let c = base_chroma * (rng.range(0.75, 1.25) as f32);
            (Oklab { l, a: c * hue.cos() as f32, b: c * hue.sin() as f32 }, MAX16)
        })
        .collect()
}

/// Resolves a generator's palette: the caller's colors when given, otherwise a
/// generated one with `n` entries.
fn resolve_palette(
    colors: &Option<Vec<[u16; 4]>>,
    n: usize,
    rng: &mut SplitMix64,
) -> Vec<(Oklab, f32)> {
    match colors {
        Some(list) if !list.is_empty() => list
            .iter()
            .map(|c| {
                let (lab, a) = rgba16_to_oklab(*c);
                (lab, a as f32)
            })
            .collect(),
        _ => random_palette(n, rng),
    }
}

/// Samples a palette as a continuous ramp at `t` in [0, 1].
#[inline]
fn palette_sample(palette: &[(Oklab, f32)], t: f64) -> (Oklab, f32) {
    match palette.len() {
        0 => (Oklab { l: 0.5, a: 0.0, b: 0.0 }, MAX16),
        1 => palette[0],
        n => {
            let scaled = t.clamp(0.0, 1.0) * (n - 1) as f64;
            let i = (scaled.floor() as usize).min(n - 2);
            let f = (scaled - i as f64) as f32;
            let (c0, a0) = palette[i];
            let (c1, a1) = palette[i + 1];
            (
                Oklab {
                    l: c0.l + (c1.l - c0.l) * f,
                    a: c0.a + (c1.a - c0.a) * f,
                    b: c0.b + (c1.b - c0.b) * f,
                },
                a0 + (a1 - a0) * f,
            )
        }
    }
}

/// The darkest entry in a palette, used as a backdrop by the generators that
/// draw light marks on a dark field.
fn palette_darkest(palette: &[(Oklab, f32)]) -> Oklab {
    palette
        .iter()
        .map(|(c, _)| *c)
        .min_by(|a, b| a.l.partial_cmp(&b.l).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(Oklab { l: 0.15, a: 0.0, b: 0.0 })
}

/// Deterministic 2D -> 64-bit hash (the SplitMix64 finalizer over a mixed
/// key). Used instead of a stateful PRNG so any pixel can look up any cell's
/// data directly, in any order, from any thread.
#[inline]
fn hash2d(ix: i64, iy: i64, seed: u64) -> u64 {
    let mut z = (ix as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (iy as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
        ^ seed;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Two independent floats in [0.0, 1.0) sliced out of one hash.
#[inline]
fn hash_unit2(h: u64) -> (f64, f64) {
    const SCALE: f64 = 1.0 / (1u64 << 26) as f64;
    ((h >> 38) as f64 * SCALE, ((h >> 12) & 0x03FF_FFFF) as f64 * SCALE)
}

/// Resolves a generator seed, reporting an auto-chosen one on stderr so the
/// user can capture an interesting output and reproduce it with `seed:N`.
fn resolve_seed(seed: &Option<u64>, what: &str) -> u64 {
    match seed {
        Some(s) => *s,
        None => {
            let s = random_seed();
            eprintln!("canvas {}: using auto seed {}", what, s);
            s
        }
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

        Self { spine_a, spine_b, inv_max_dist, t_min, t_max, shape }
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

        let shaped = bezier1d(self.shape[0], self.shape[1], self.shape[2], self.shape[3], d_norm)
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

/// Radius, in cell widths, at which a blob cell's influence reaches exactly
/// zero. This is what makes the finite search window correct: a seed outside
/// the searched 5x5 block is always more than 2 cell widths away, so it would
/// contribute nothing even if it were included.
const BLOB_SUPPORT: f64 = 2.0;

/// Voronoi cells over a jittered grid.
///
/// Both styles search a fixed block of grid cells around the pixel, which is
/// what keeps this O(1) per pixel. The block has to be wide enough that no
/// excluded seed could have changed the answer, otherwise seeds pop in and out
/// as the block slides and the grid pitch shows up in the output:
///
/// * `sharp` needs the true nearest seed. A seed outside the 3x3 block is more
///   than 1 cell away, so a 3x3 result under that distance is already provably
///   correct; only when it isn't do we widen to 5x5.
/// * `blob` sums every seed with non-zero weight, so its window must cover the
///   whole kernel support -- hence 5x5 paired with `BLOB_SUPPORT`.
#[allow(clippy::too_many_arguments)]
fn render_voronoi(
    buf: &mut [u16],
    width: u32,
    height: u32,
    seed: u64,
    cells: u32,
    style: VoronoiStyle,
    softness: f64,
    palette: &[(Oklab, f32)],
) {
    let row_len = width as usize * 4;
    // Cell pitch comes off the short edge so cells stay square on a
    // non-square canvas instead of stretching with it.
    let short = width.min(height).max(1) as f64;
    let cell = (short / cells.max(1) as f64).max(1.0);
    let inv_cell = 1.0 / cell;
    let pal_n = palette.len().max(1);
    // Exponential falloff constant, expressed in cell widths.
    let falloff = 1.0 / softness.max(1e-3);

    buf.par_chunks_mut(row_len).enumerate().for_each(|(y, row)| {
        let py = y as f64 + 0.5;
        let gy = (py * inv_cell).floor() as i64;
        for (x, px) in row.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let pxf = x as f64 + 0.5;
            let gx = (pxf * inv_cell).floor() as i64;

            // Distance from this pixel to the seed of one grid cell, in
            // cell widths so `softness` means the same thing at every
            // cell count.
            let seed_dist = |ox: i64, oy: i64| -> (f64, u64) {
                let (cx, cy) = (gx + ox, gy + oy);
                let h = hash2d(cx, cy, seed);
                let (jx, jy) = hash_unit2(h);
                let dx = (cx as f64 + jx) * cell - pxf;
                let dy = (cy as f64 + jy) * cell - py;
                ((dx * dx + dy * dy).sqrt() * inv_cell, h)
            };

            // The palette index uses the low hash bits, which are
            // independent of the bits `hash_unit2` spends on jitter.
            let (lab, alpha) = match style {
                VoronoiStyle::Sharp => {
                    let nearest = |radius: i64| -> (f64, u64) {
                        let mut best = (f64::INFINITY, 0_u64);
                        for oy in -radius..=radius {
                            for ox in -radius..=radius {
                                let cand = seed_dist(ox, oy);
                                if cand.0 < best.0 {
                                    best = cand;
                                }
                            }
                        }
                        best
                    };
                    let mut best = nearest(1);
                    // The pixel's own cell always holds a seed within
                    // sqrt(2) cells, so this widening is rare.
                    if best.0 > 1.0 {
                        best = nearest(2);
                    }
                    palette[(best.1 & 0xFFF) as usize % pal_n]
                }
                VoronoiStyle::Blob => {
                    let mut seeds = [(0.0_f64, 0_u64); 25];
                    let mut count = 0_usize;
                    let mut d_min = f64::INFINITY;
                    for oy in -2..=2_i64 {
                        for ox in -2..=2_i64 {
                            let (d, h) = seed_dist(ox, oy);
                            if d < BLOB_SUPPORT {
                                seeds[count] = (d, h);
                                count += 1;
                                if d < d_min {
                                    d_min = d;
                                }
                            }
                        }
                    }

                    let (mut l, mut a, mut b, mut al, mut wsum) =
                        (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
                    for &(d, h) in seeds.iter().take(count) {
                        // Wyvill's kernel windows the exponential falloff
                        // down to exactly zero (with zero slope) at the
                        // support radius, so a cell's contribution fades
                        // out smoothly instead of being cut off at the
                        // edge of the search block.
                        let t = d * (1.0 / BLOB_SUPPORT);
                        let window = 1.0 - t * t;
                        let window = window * window * window;
                        // Offsetting by d_min keeps exp() off its
                        // underflow floor; it is a common factor across
                        // every weight, so it cancels in the normalization
                        // and cannot affect the result.
                        let w = (-(d - d_min) * falloff).exp() * window;
                        let c = palette[(h & 0xFFF) as usize % pal_n];
                        l += w * c.0.l as f64;
                        a += w * c.0.a as f64;
                        b += w * c.0.b as f64;
                        al += w * c.1 as f64;
                        wsum += w;
                    }

                    if wsum > 1e-12 {
                        let inv = 1.0 / wsum;
                        (
                            Oklab { l: (l * inv) as f32, a: (a * inv) as f32, b: (b * inv) as f32 },
                            (al * inv) as f32,
                        )
                    } else {
                        // Unreachable in practice -- the pixel's own cell
                        // always holds a seed inside the support radius --
                        // but this keeps a zero divide from painting the
                        // pixel black, and cannot panic on an empty slice.
                        palette
                            .first()
                            .copied()
                            .unwrap_or((Oklab { l: 0.0, a: 0.0, b: 0.0 }, MAX16))
                    }
                }
            };

            let rgb = oklab_to_rgb16(lab);
            px.copy_from_slice(&[rgb[0], rgb[1], rgb[2], alpha.round().clamp(0.0, MAX16) as u16]);
        }
    });
}

/// Fractal Brownian motion from the `noise` crate, mapped through a ramp.
#[allow(clippy::too_many_arguments)]
fn render_fbm(
    buf: &mut [u16],
    width: u32,
    height: u32,
    seed: u64,
    octaves: usize,
    frequency: f64,
    lacunarity: f64,
    persistence: f64,
    zoom: f64,
    seamless: bool,
    grad: &PreparedGradient,
    easing: Option<&CssEasing>,
) {
    // Order matters: `set_persistence` recomputes the internal scale factor
    // from the octave count, so octaves has to be set first.
    // `noise` seeds are u32, so the canvas seed is deliberately narrowed here.
    let fbm = Fbm::<Perlin>::new(seed as u32)
        .set_octaves(octaves.clamp(1, Fbm::<Perlin>::MAX_OCTAVES))
        .set_frequency(frequency)
        .set_lacunarity(lacunarity)
        .set_persistence(persistence);

    // Keep the field isotropic: the short edge spans `zoom` units and the long
    // edge is extended in proportion rather than stretching the noise.
    let (w, h) = (width as f64, height as f64);
    let (ex, ey) = if w >= h { (zoom * w / h, zoom) } else { (zoom, zoom * h / w) };

    // `new_fn` rather than `new`: in noise 0.9 the 2-D `build` is only
    // implemented for a builder whose source went through `new_fn`, so
    // `PlaneMapBuilder::<_, 2>::new(&fbm)` (as the crate README shows) has no
    // `set_size` or `build` to call. The turbofish pins the dimension rather
    // than leaving it to be inferred through the closure's `Fn` bound.
    let map = PlaneMapBuilder::<_, 2>::new_fn(|p: [f64; 2]| fbm.get(p))
        .set_size(width as usize, height as usize)
        .set_x_bounds(-ex, ex)
        .set_y_bounds(-ey, ey)
        .set_is_seamless(seamless)
        .build();

    let row_len = width as usize * 4;
    let map = &map;
    buf.par_chunks_mut(row_len).enumerate().for_each(|(y, row)| {
        for (x, px) in row.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            // Fbm is scaled to [-1, 1]; remap onto the ramp's [0, 1].
            let mut t = ((map.get_value(x, y) + 1.0) * 0.5).clamp(0.0, 1.0);
            if let Some(e) = easing {
                t = e.ease(t);
            }
            px.copy_from_slice(&grad.eval(t));
        }
    });
}

/// Deposits one sub-pixel sample into a coverage buffer with bilinear weights.
/// Colors are accumulated premultiplied by coverage so overlapping strands
/// average rather than fight, independent of the order they were drawn in.
#[inline]
fn splat(acc: &mut [[f32; 4]], width: u32, height: u32, x: f64, y: f64, c: Oklab, a: f32) {
    if a <= 0.0 {
        return;
    }
    let fx = x - 0.5;
    let fy = y - 0.5;
    let bx = fx.floor();
    let by = fy.floor();
    let tx = (fx - bx) as f32;
    let ty = (fy - by) as f32;
    let x0 = bx as i64;
    let y0 = by as i64;
    let corners = [
        (0_i64, 0_i64, (1.0 - tx) * (1.0 - ty)),
        (1, 0, tx * (1.0 - ty)),
        (0, 1, (1.0 - tx) * ty),
        (1, 1, tx * ty),
    ];
    for (dx, dy, wgt) in corners {
        let px = x0 + dx;
        let py = y0 + dy;
        if px < 0 || py < 0 || px >= width as i64 || py >= height as i64 {
            continue;
        }
        let cw = wgt * a;
        if cw <= 0.0 {
            continue;
        }
        let cell = &mut acc[py as usize * width as usize + px as usize];
        cell[0] += cw * c.l;
        cell[1] += cw * c.a;
        cell[2] += cw * c.b;
        cell[3] += cw;
    }
}

/// Hair-like streamlines: particles seeded on a jittered grid and advected
/// through a Perlin flow field, tapering at both ends so they read as strands
/// rather than tubes.
#[allow(clippy::too_many_arguments)]
fn render_flow(
    buf: &mut [u16],
    width: u32,
    height: u32,
    seed: u64,
    strands: u32,
    steps: u32,
    step_len: f64,
    zoom: f64,
    turns: f64,
    line_width: f64,
    alpha: f64,
    palette: &[(Oklab, f32)],
) {
    use std::f64::consts::PI;

    let (w, h) = (width as f64, height as f64);
    let mut acc: Vec<[f32; 4]> = vec![[0.0; 4]; width as usize * height as usize];

    let field =
        Fbm::<Perlin>::new(seed as u32).set_octaves(4).set_frequency(1.0).set_persistence(0.5);

    // Strand starts sit on a jittered grid: pure random placement clumps, and
    // clumps show up badly once every strand is a visible line.
    let target = strands.max(1) as f64;
    let cols = ((target * w / h).sqrt().round() as i64).max(1);
    let rows = ((target * h / w).sqrt().round() as i64).max(1);

    let short = w.min(h).max(1.0);
    let noise_scale = zoom.max(1e-6) / short;
    let sub = line_width.max(0.5).ceil().max(1.0) as i64;
    let sub_gap = line_width.max(0.5) / sub as f64;
    let a_step = alpha.clamp(0.0, 1.0) as f32;
    let steps = steps.max(1);

    for gy in 0..rows {
        for gx in 0..cols {
            let h0 = hash2d(gx, gy, seed);
            let (jx, jy) = hash_unit2(h0);
            let mut x = (gx as f64 + jx) / cols as f64 * w;
            let mut y = (gy as f64 + jy) / rows as f64 * h;

            // Color by start position so neighbouring strands share a hue and
            // the field reads as regions rather than confetti.
            let t = (x / w) * 0.65 + (y / h) * 0.35;
            let (base, _) = palette_sample(palette, t);
            let jitter = ((h0 >> 20) & 0xFFFF) as f32 / 65535.0 - 0.5;
            let col = Oklab { l: (base.l + jitter * 0.12).clamp(0.0, 1.0), a: base.a, b: base.b };

            for s in 0..steps {
                let ang = field.get([x * noise_scale, y * noise_scale]) * PI * turns;
                let (dy_dir, dx_dir) = ang.sin_cos();
                x += dx_dir * step_len;
                y += dy_dir * step_len;
                if x < -2.0 || y < -2.0 || x > w + 2.0 || y > h + 2.0 {
                    break;
                }
                // Fade in and out along the strand so the ends taper.
                let u = (s as f64 + 0.5) / steps as f64;
                let env = (PI * u).sin().sqrt() as f32;
                let a = a_step * env;
                for k in 0..sub {
                    let off = (k as f64 - (sub - 1) as f64 * 0.5) * sub_gap;
                    // Offset perpendicular to travel to give the strand width.
                    splat(&mut acc, width, height, x - dy_dir * off, y + dx_dir * off, col, a);
                }
            }
        }
    }

    // Backdrop: the palette's darkest color, pushed darker still so even the
    // dimmest strand stays legible against it.
    let dark = palette_darkest(palette);
    let bg = Oklab { l: (dark.l * 0.55).clamp(0.02, 0.5), a: dark.a * 0.5, b: dark.b * 0.5 };

    let row_len = width as usize * 4;
    let acc = &acc;
    buf.par_chunks_mut(row_len).enumerate().for_each(|(y, row)| {
        for (x, px) in row.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let cell = acc[y * width as usize + x];
            let cov = cell[3];
            let lab = if cov > 1e-6 {
                let inv = 1.0 / cov;
                let a = cov.min(1.0);
                Oklab {
                    l: bg.l + (cell[0] * inv - bg.l) * a,
                    a: bg.a + (cell[1] * inv - bg.a) * a,
                    b: bg.b + (cell[2] * inv - bg.b) * a,
                }
            } else {
                bg
            };
            let rgb = oklab_to_rgb16(lab);
            px.copy_from_slice(&[rgb[0], rgb[1], rgb[2], u16::MAX]);
        }
    });
}

/// Low-poly facets: a Delaunay triangulation of jittered points, each triangle
/// filled from a smooth underlying color field.
fn render_lowpoly(
    buf: &mut [u16],
    width: u32,
    height: u32,
    seed: u64,
    points: u32,
    smooth: bool,
    palette: &[(Oklab, f32)],
) {
    let (w, h) = (width as f64, height as f64);
    let mut rng = SplitMix64::new(seed);

    // Interior points on a jittered grid. Uniform random points leave clumps
    // and slivers; jittered grid points give evenly sized facets.
    let target = points.max(3) as f64;
    let aspect = (w / h).max(1e-6);
    let cols = ((target * aspect).sqrt().round() as usize).max(2);
    let rows = ((target / aspect).sqrt().round() as usize).max(2);

    let mut pts: Vec<DelaunayPoint> = Vec::with_capacity(cols * rows + 4 * (cols + rows) + 8);
    for gy in 0..rows {
        for gx in 0..cols {
            let jx = rng.range(0.12, 0.88);
            let jy = rng.range(0.12, 0.88);
            pts.push(DelaunayPoint {
                x: (gx as f64 + jx) / cols as f64 * w,
                y: (gy as f64 + jy) / rows as f64 * h,
            });
        }
    }

    // Boundary points, so the convex hull of the input is the canvas rectangle
    // and every pixel ends up inside some triangle. Corners are contributed
    // once by the horizontal edges; the vertical edges skip them to avoid
    // handing the triangulator duplicate points.
    let edge_x = cols.max(2);
    let edge_y = rows.max(2);
    for i in 0..=edge_x {
        let t = i as f64 / edge_x as f64;
        pts.push(DelaunayPoint { x: t * w, y: 0.0 });
        pts.push(DelaunayPoint { x: t * w, y: h });
    }
    for i in 1..edge_y {
        let t = i as f64 / edge_y as f64;
        pts.push(DelaunayPoint { x: 0.0, y: t * h });
        pts.push(DelaunayPoint { x: w, y: t * h });
    }

    // Underlying color field: a bilinear Oklab blend of four palette entries,
    // so adjacent facets differ subtly instead of at random.
    let corner = |i: usize| palette_sample(palette, i as f64 / 3.0).0;
    let (c_tl, c_tr, c_bl, c_br) = (corner(0), corner(1), corner(2), corner(3));
    let field = |x: f64, y: f64| -> Oklab {
        let u = (x / w).clamp(0.0, 1.0) as f32;
        let v = (y / h).clamp(0.0, 1.0) as f32;
        let (ui, vi) = (1.0 - u, 1.0 - v);
        let (w00, w10, w01, w11) = (ui * vi, u * vi, ui * v, u * v);
        Oklab {
            l: w00 * c_tl.l + w10 * c_tr.l + w01 * c_bl.l + w11 * c_br.l,
            a: w00 * c_tl.a + w10 * c_tr.a + w01 * c_bl.a + w11 * c_br.a,
            b: w00 * c_tl.b + w10 * c_tr.b + w01 * c_bl.b + w11 * c_br.b,
        }
    };

    let tri = triangulate(&pts);
    let width_us = width as usize;

    // Collinear input degenerates to an empty triangulation; fall back to the
    // bare color field rather than leaving the canvas blank.
    if tri.triangles.is_empty() {
        let row_len = width_us * 4;
        buf.par_chunks_mut(row_len).enumerate().for_each(|(y, row)| {
            for (x, px) in row.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                let rgb = oklab_to_rgb16(field(x as f64 + 0.5, y as f64 + 0.5));
                px.copy_from_slice(&[rgb[0], rgb[1], rgb[2], u16::MAX]);
            }
        });
        return;
    }

    for t in 0..tri.len() {
        let (ia, ib, ic) =
            (tri.triangles[3 * t], tri.triangles[3 * t + 1], tri.triangles[3 * t + 2]);
        let (pa, pb, pc) = (&pts[ia], &pts[ib], &pts[ic]);

        // Twice the signed area. Dividing the barycentric weights by it also
        // normalizes the winding, so the inside test works either way round.
        let area2 = (pb.x - pa.x) * (pc.y - pa.y) - (pb.y - pa.y) * (pc.x - pa.x);
        if area2.abs() < 1e-12 {
            continue;
        }
        let inv_area = 1.0 / area2;

        // Per-facet lightness offset, keyed off the triangle's own position so
        // it stays stable for a given seed. This is what sells the "faceted
        // 3-D surface" read; without it the mesh looks like a flat gradient.
        let cx = (pa.x + pb.x + pc.x) / 3.0;
        let cy = (pa.y + pb.y + pc.y) / 3.0;
        let facet = (hash2d(cx as i64, cy as i64, seed) >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
        let shade = facet * 0.09;

        let (ca, cb, cc) = if smooth {
            (field(pa.x, pa.y), field(pb.x, pb.y), field(pc.x, pc.y))
        } else {
            let flat = field(cx, cy);
            (flat, flat, flat)
        };

        let min_x = pa.x.min(pb.x).min(pc.x).floor().max(0.0) as usize;
        let max_x = (pa.x.max(pb.x).max(pc.x).ceil() as i64).clamp(0, width as i64 - 1) as usize;
        let min_y = pa.y.min(pb.y).min(pc.y).floor().max(0.0) as usize;
        let max_y = (pa.y.max(pb.y).max(pc.y).ceil() as i64).clamp(0, height as i64 - 1) as usize;
        if min_x > max_x || min_y > max_y {
            continue;
        }

        for py in min_y..=max_y {
            let fy = py as f64 + 0.5;
            for px_i in min_x..=max_x {
                let fx = px_i as f64 + 0.5;
                // Barycentric weights: wb for B, wc for C, wa the remainder.
                let wb = ((fx - pa.x) * (pc.y - pa.y) - (fy - pa.y) * (pc.x - pa.x)) * inv_area;
                let wc = ((pb.x - pa.x) * (fy - pa.y) - (pb.y - pa.y) * (fx - pa.x)) * inv_area;
                let wa = 1.0 - wb - wc;
                // A hair of slack on the edges: neighbouring triangles then
                // overlap by a sliver instead of leaving a seam of unwritten
                // pixels where the two tests disagree by a rounding error.
                if wa < -1e-9 || wb < -1e-9 || wc < -1e-9 {
                    continue;
                }

                let lab = if smooth {
                    let (fa, fb, fc) = (wa as f32, wb as f32, wc as f32);
                    Oklab {
                        l: (ca.l * fa + cb.l * fb + cc.l * fc + shade).clamp(0.0, 1.0),
                        a: ca.a * fa + cb.a * fb + cc.a * fc,
                        b: ca.b * fa + cb.b * fb + cc.b * fc,
                    }
                } else {
                    Oklab { l: (ca.l + shade).clamp(0.0, 1.0), a: ca.a, b: ca.b }
                };

                let rgb = oklab_to_rgb16(lab);
                let o = (py * width_us + px_i) * 4;
                buf[o] = rgb[0];
                buf[o + 1] = rgb[1];
                buf[o + 2] = rgb[2];
                buf[o + 3] = u16::MAX;
            }
        }
    }
}

/// Hard ceiling on chaos-game iterations, whatever `quality` asks for. The
/// chaos game is a serial random scatter, so this is the knob that decides how
/// long a `flame` canvas can take; raising it costs time roughly linearly.
const MAX_FLAME_ITERS: u64 = 400_000_000;

/// Variations that stay continuous across the `atan2` branch cut along the
/// negative y axis. A variation that jumps there folds the plane along that
/// line, which renders as a hard straight edge through the image as soon as
/// the transform carries real weight. `continuous` draws from this set only.
///
/// Using `theta` is not by itself disqualifying: most of these feed it through
/// `sin`/`cos`, which are 2*pi-periodic, so the 2*pi jump at the cut cancels
/// out. Only three genuinely break -- polar (5) and disc (8) use `theta / PI`
/// raw, and heart (7) forms `sin(theta * r)`, which is periodic in `theta`
/// only when `r` happens to be an integer.
const CONTINUOUS_VARIATIONS: [u8; 14] = [0, 1, 2, 3, 4, 6, 9, 10, 11, 12, 13, 14, 15, 16];

/// One affine map plus its variation, i.e. one function of the iterated
/// function system.
struct FlameTransform {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    e: f64,
    f: f64,
    /// Where this transform sits on the palette, in [0, 1].
    color: f64,
    variation: u8,
}

/// A representative subset of the Draves variation set. `theta` follows the
/// flame convention of `atan2(x, y)` rather than the usual `atan2(y, x)`.
#[inline]
fn flame_variation(v: u8, x: f64, y: f64) -> (f64, f64) {
    use std::f64::consts::PI;

    let r2 = x * x + y * y;
    let r = r2.sqrt();
    let theta = x.atan2(y);
    let inv_r = 1.0 / (r + 1e-9);

    match v {
        0 => (x, y),                             // linear
        1 => (x.sin(), y.sin()),                 // sinusoidal
        2 => (x / (r2 + 1e-9), y / (r2 + 1e-9)), // spherical
        3 => {
            let (s, c) = r2.sin_cos();
            (x * s - y * c, x * c + y * s) // swirl
        }
        4 => ((x - y) * (x + y) * inv_r, 2.0 * x * y * inv_r), // horseshoe
        5 => (theta / PI, r - 1.0),                            // polar
        6 => (r * (theta + r).sin(), r * (theta - r).cos()),   // handkerchief
        7 => (r * (theta * r).sin(), -r * (theta * r).cos()),  // heart
        8 => {
            let t = theta / PI;
            let (s, c) = (PI * r).sin_cos();
            (t * s, t * c) // disc
        }
        9 => (inv_r * (theta.cos() + r.sin()), inv_r * (theta.sin() - r.cos())), // spiral
        10 => (theta.sin() * inv_r, r * theta.cos()),                            // hyperbolic
        11 => (theta.sin() * r.cos(), theta.cos() * r.sin()),                    // diamond
        // 12..=16 are the cut-free additions. They exist so `continuous` has a
        // pool worth drawing from; the numbering starts above the originals so
        // that the default `% 12` selection, and every seed rendered with it,
        // is left exactly as it was.
        12 => {
            let s = 2.0 / (r + 1.0);
            (s * y, s * x) // fisheye
        }
        13 => {
            let s = 2.0 / (r + 1.0);
            (s * x, s * y) // eyefish
        }
        14 => {
            // Clamped so a far-flung point cannot overflow to infinity; the
            // iteration guard would catch it, but a finite value keeps the
            // orbit usable instead of forcing a restart.
            let m = (x - 1.0).min(80.0).exp();
            (m * (PI * y).cos(), m * (PI * y).sin()) // exponential
        }
        15 => {
            let s = 4.0 / (r2 + 4.0);
            (s * x, s * y) // bubble
        }
        16 => (x.sin(), y), // cylinder
        _ => (x, y),
    }
}

/// Applies one transform: affine first, then its variation.
#[inline]
fn flame_step(t: &FlameTransform, x: f64, y: f64) -> (f64, f64) {
    flame_variation(t.variation, t.a * x + t.b * y + t.c, t.d * x + t.e * y + t.f)
}

/// Fractal flame: the chaos game scattered into a density histogram, then
/// log-tone-mapped. Accumulation is single-threaded on purpose -- a per-thread
/// histogram would cost 16 bytes per pixel per thread, which is a worse trade
/// than the time saved on a scatter this cache-hostile.
#[allow(clippy::too_many_arguments)]
fn render_flame(
    buf: &mut [u16],
    width: u32,
    height: u32,
    seed: u64,
    quality: u32,
    transforms: u32,
    gamma: f64,
    continuous: bool,
    palette: &[(Oklab, f32)],
) {
    let mut rng = SplitMix64::new(seed);
    let n_t = transforms.clamp(2, 12) as usize;

    // Random affine maps, biased toward contraction so the attractor usually
    // stays bounded. Divergent draws are caught during iteration anyway.
    let mut xf: Vec<FlameTransform> = Vec::with_capacity(n_t);
    for i in 0..n_t {
        let scale = rng.range(0.3, 0.9);
        let (sn, cs) = rng.range(0.0, std::f64::consts::TAU).sin_cos();
        xf.push(FlameTransform {
            a: scale * cs,
            b: -scale * sn * rng.range(0.6, 1.4),
            c: rng.range(-1.0, 1.0),
            d: scale * sn,
            e: scale * cs * rng.range(0.6, 1.4),
            f: rng.range(-1.0, 1.0),
            color: if n_t > 1 { i as f64 / (n_t - 1) as f64 } else { 0.0 },
            // One draw either way, so switching `continuous` on does not
            // shift the PRNG stream for anything that follows.
            variation: if continuous {
                CONTINUOUS_VARIATIONS
                    [(rng.next_u64() % CONTINUOUS_VARIATIONS.len() as u64) as usize]
            } else {
                (rng.next_u64() % 12) as u8
            },
        });
    }

    // Palette in linear light, so summing samples is physically sensible.
    let lut: Vec<[f32; 3]> = (0..256)
        .map(|i| {
            let (lab, _) = palette_sample(palette, i as f64 / 255.0);
            let lin = oklab_to_linear_srgb(lab);
            [lin.r.max(0.0), lin.g.max(0.0), lin.b.max(0.0)]
        })
        .collect();

    // --- pass 1: sample the attractor to work out where to point the camera.
    let mut xs: Vec<f64> = Vec::with_capacity(40_000);
    let mut ys: Vec<f64> = Vec::with_capacity(40_000);
    let (mut x, mut y) = (rng.range(-1.0, 1.0), rng.range(-1.0, 1.0));
    for i in 0..40_000_u32 {
        let t = &xf[(rng.next_u64() % n_t as u64) as usize];
        let (nx, ny) = flame_step(t, x, y);
        x = nx;
        y = ny;
        if !x.is_finite() || !y.is_finite() || x.abs() > 1e6 || y.abs() > 1e6 {
            x = rng.range(-1.0, 1.0);
            y = rng.range(-1.0, 1.0);
            continue;
        }
        if i > 20 {
            xs.push(x);
            ys.push(y);
        }
    }

    // Percentile bounds rather than min/max: a handful of far-flung outliers
    // would otherwise shrink the whole attractor to a dot in the middle.
    let bounds = |v: &mut Vec<f64>| -> (f64, f64) {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let lo = v[v.len() / 200];
        let hi = v[v.len() - 1 - v.len() / 200];
        if hi - lo < 1e-9 { (lo - 1.0, hi + 1.0) } else { (lo, hi) }
    };

    let (cam_cx, cam_cy, scale_x, scale_y) = if xs.len() < 400 {
        // Degenerate system: frame the default unit square and let the
        // background carry the image.
        (0.0, 0.0, width as f64 * 0.5, height as f64 * 0.5)
    } else {
        let (x0, x1) = bounds(&mut xs);
        let (y0, y1) = bounds(&mut ys);
        let mut sx = (x1 - x0) * 0.55;
        let mut sy = (y1 - y0) * 0.55;
        // Match the canvas aspect so the attractor is not squashed.
        let aspect = width as f64 / height as f64;
        if sx / sy < aspect {
            sx = sy * aspect;
        } else {
            sy = sx / aspect;
        }
        ((x0 + x1) * 0.5, (y0 + y1) * 0.5, width as f64 / (2.0 * sx), height as f64 / (2.0 * sy))
    };

    // --- pass 2: the real chaos game.
    //
    // `quality` is samples per pixel, so the work grows with canvas area and
    // has to be bounded somewhere. Silently clamping made two very different
    // `quality:` values render byte-for-byte identically, so the clamp is
    // reported: past this point the only thing more samples buy is less
    // Monte-Carlo noise, since the tone map normalizes by the peak density.
    let px_total = (width as u64).saturating_mul(height as u64).max(1);
    let requested = (quality as u64).saturating_mul(px_total);
    let iters = requested.min(MAX_FLAME_ITERS);
    if iters < requested {
        eprintln!(
            "canvas flame: quality {} needs {} iterations at {}x{}, over the {}M ceiling; \
             rendering at quality {} instead",
            quality,
            requested,
            width,
            height,
            MAX_FLAME_ITERS / 1_000_000,
            (iters / px_total).max(1)
        );
    }

    let width_us = width as usize;
    let mut hist: Vec<[f32; 4]> = vec![[0.0; 4]; width_us * height as usize];

    let (mut x, mut y) = (rng.range(-1.0, 1.0), rng.range(-1.0, 1.0));
    let mut col = rng.next_f64();
    let half_w = width as f64 * 0.5;
    let half_h = height as f64 * 0.5;

    for i in 0..iters {
        let t = &xf[(rng.next_u64() % n_t as u64) as usize];
        let (nx, ny) = flame_step(t, x, y);
        x = nx;
        y = ny;
        // Colour drifts halfway toward the chosen transform's index each step,
        // which is what ties a region's hue to the path that reached it.
        col = (col + t.color) * 0.5;
        if !x.is_finite() || !y.is_finite() || x.abs() > 1e6 || y.abs() > 1e6 {
            x = rng.range(-1.0, 1.0);
            y = rng.range(-1.0, 1.0);
            col = rng.next_f64();
            continue;
        }
        // Discard the first few points: they are still settling onto the
        // attractor and would smear the image with off-shape samples.
        if i < 20 {
            continue;
        }

        let sx = (x - cam_cx) * scale_x + half_w;
        let sy = (y - cam_cy) * scale_y + half_h;
        if sx < 0.0 || sy < 0.0 {
            continue;
        }
        let (ix, iy) = (sx as usize, sy as usize);
        if ix >= width_us || iy >= height as usize {
            continue;
        }

        let rgb = lut[(col.clamp(0.0, 1.0) * 255.0) as usize];
        let cell = &mut hist[iy * width_us + ix];
        cell[0] += rgb[0];
        cell[1] += rgb[1];
        cell[2] += rgb[2];
        cell[3] += 1.0;
    }

    // --- tone map. Brightness follows log density, which is what stops the
    // dense core from blowing out while the faint filaments stay visible.
    let max_count = hist.iter().fold(0.0_f32, |m, h| m.max(h[3]));
    let denom = (1.0 + max_count).ln().max(1e-6);
    let inv_gamma = 1.0 / (gamma.max(0.1) as f32);

    let dark = palette_darkest(palette);
    let bg_lin = oklab_to_linear_srgb(Oklab {
        l: (dark.l * 0.30).clamp(0.01, 0.4),
        a: dark.a * 0.4,
        b: dark.b * 0.4,
    });

    let row_len = width_us * 4;
    let hist = &hist;
    buf.par_chunks_mut(row_len).enumerate().for_each(|(y, row)| {
        for (x, px) in row.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let cell = hist[y * width_us + x];
            let cnt = cell[3];
            let (r, g, b) = if cnt > 0.0 {
                let a = ((1.0 + cnt).ln() / denom).powf(inv_gamma).clamp(0.0, 1.0);
                let inv = 1.0 / cnt;
                (
                    cell[0] * inv * a + bg_lin.r * (1.0 - a),
                    cell[1] * inv * a + bg_lin.g * (1.0 - a),
                    cell[2] * inv * a + bg_lin.b * (1.0 - a),
                )
            } else {
                (bg_lin.r, bg_lin.g, bg_lin.b)
            };
            let rgb = linear_rgb_to_rgb16(r, g, b);
            px.copy_from_slice(&[rgb[0], rgb[1], rgb[2], u16::MAX]);
        }
    });
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

    // Length in *samples*, not bytes: the buffer is 16 bits per channel.
    // The extra `checked_mul` makes sure the byte size fits too, so we report a
    // clean error instead of letting `Vec` abort on capacity overflow.
    let buf_len = (width as usize)
        .checked_mul(height as usize)
        .and_then(|p| p.checked_mul(4))
        .filter(|len| len.checked_mul(std::mem::size_of::<u16>()).is_some())
        .ok_or_else(|| crate::wm_err!("canvas: image dimensions overflow usize"))?;

    let mut buf: Vec<u16> = vec![0; buf_len];
    let row_len = width as usize * 4;

    match &config.spec {
        CanvasSpec::Solid(c) => {
            // Parallel fill. chunks_exact_mut of size 4 is faster than .copy_from_slice on
            // the whole row because the optimiser turns it into a memset-friendly loop.
            buf.par_chunks_mut(row_len).for_each(|row| {
                for px in row.as_chunks_mut::<4>().0 {
                    px.copy_from_slice(c);
                }
            });
        }
        CanvasSpec::Linear { angle_deg, stops, easing } => {
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

            buf.par_chunks_mut(row_len).enumerate().for_each(|(y, row)| {
                let dy = y as f64 - cy;
                let dy_sin = dy * sin_t;
                for (x, px) in row.as_chunks_mut::<4>().0.iter_mut().enumerate() {
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
        CanvasSpec::Radial { center_x, center_y, stops, easing } => {
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

            buf.par_chunks_mut(row_len).enumerate().for_each(|(y, row)| {
                let dy = y as f64 - cy;
                let dy2 = dy * dy;
                for (x, px) in row.as_chunks_mut::<4>().0.iter_mut().enumerate() {
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

            buf.par_chunks_mut(row_len).enumerate().for_each(|(y, row)| {
                let v = y as f64 / height_f;
                for (x, px) in row.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                    let u = x as f64 / width_f;
                    let c = colors.eval_color(u, v);
                    px.copy_from_slice(&c);
                }
            });
        }
        CanvasSpec::Coons { seed, colors, easing, transparency } => {
            let w = width as f64;
            let h = height as f64;

            // Resolve the seed: use the caller's or derive one from OS entropy.
            // When auto-seeding, report the chosen seed on stderr so the user
            // can capture an interesting output and reproduce it with `seed:N`.
            let actual_seed = resolve_seed(seed, "coons");
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

            buf.par_chunks_mut(row_len).enumerate().for_each(|(y, row)| {
                let v_fallback = y as f64 * inv_h;
                let y_f = y as f64;
                for (x, px) in row.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                    let pt = Point::new(x as f64, y_f);

                    // Solve for (u, v) via Newton-Raphson; fall back to
                    // bilinear if the solver bails, guaranteeing full
                    // coverage — critical when this canvas is a background.
                    let (mut u, mut v) =
                        patch.inverse_eval(pt).unwrap_or((x as f64 * inv_w, v_fallback));

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
                        c[3] = (c[3] as f32 * alpha_factor).round().clamp(0.0, MAX16) as u16;
                    }

                    px.copy_from_slice(&c);
                }
            });
        }
        CanvasSpec::Voronoi { seed, cells, style, softness, colors } => {
            let actual_seed = resolve_seed(seed, "voronoi");
            let mut rng = SplitMix64::new(actual_seed);
            let palette = resolve_palette(colors, 6, &mut rng);
            render_voronoi(
                &mut buf,
                width,
                height,
                actual_seed,
                *cells,
                *style,
                *softness,
                &palette,
            );
        }
        CanvasSpec::Fbm {
            seed,
            octaves,
            frequency,
            lacunarity,
            persistence,
            zoom,
            seamless,
            stops,
            easing,
        } => {
            let actual_seed = resolve_seed(seed, "fbm");
            let mut rng = SplitMix64::new(actual_seed);
            let grad = match stops {
                Some(list) => PreparedGradient::new(list),
                None => {
                    // No ramp given: turn the generated palette into evenly
                    // spaced stops, staying in Oklab the whole way.
                    let palette = random_palette(4, &mut rng);
                    let last = (palette.len() - 1).max(1) as f64;
                    PreparedGradient::from_oklab(
                        palette
                            .iter()
                            .enumerate()
                            .map(|(i, (lab, a))| {
                                (i as f64 / last, *lab, a.round().clamp(0.0, MAX16) as u16)
                            })
                            .collect(),
                    )
                }
            };
            render_fbm(
                &mut buf,
                width,
                height,
                actual_seed,
                *octaves,
                *frequency,
                *lacunarity,
                *persistence,
                *zoom,
                *seamless,
                &grad,
                easing.as_ref(),
            );
        }
        CanvasSpec::Flow {
            seed,
            strands,
            steps,
            step_len,
            zoom,
            turns,
            width: line_width,
            alpha,
            colors,
        } => {
            let actual_seed = resolve_seed(seed, "flow");
            let mut rng = SplitMix64::new(actual_seed);
            let palette = resolve_palette(colors, 5, &mut rng);
            render_flow(
                &mut buf,
                width,
                height,
                actual_seed,
                *strands,
                *steps,
                *step_len,
                *zoom,
                *turns,
                *line_width,
                *alpha,
                &palette,
            );
        }
        CanvasSpec::LowPoly { seed, points, smooth, colors } => {
            let actual_seed = resolve_seed(seed, "lowpoly");
            let mut rng = SplitMix64::new(actual_seed);
            let palette = resolve_palette(colors, 4, &mut rng);
            render_lowpoly(&mut buf, width, height, actual_seed, *points, *smooth, &palette);
        }
        CanvasSpec::Flame { seed, quality, transforms, gamma, continuous, colors } => {
            let actual_seed = resolve_seed(seed, "flame");
            let mut rng = SplitMix64::new(actual_seed);
            let palette = resolve_palette(colors, 5, &mut rng);
            render_flame(
                &mut buf,
                width,
                height,
                actual_seed,
                *quality,
                *transforms,
                *gamma,
                *continuous,
                &palette,
            );
        }
    }

    let out = Rgba16Image::from_raw(width, height, buf).ok_or_else(|| {
        crate::wm_err!("canvas: failed to construct image buffer (dimensions too large?)")
    })?;

    if is_overlay {
        // We are mutating an existing image. Convert to 16-bit RGBA and
        // alpha-blend `out` OVER it. Promoting the base rather than demoting
        // the canvas keeps the gradient's extra precision through compositing.
        let mut base = image.pixels.to_rgba16();
        image::imageops::overlay(&mut base, &out, 0, 0);
        image.pixels = DynamicImage::ImageRgba16(base);
    } else {
        // We are generating a new image.
        image.pixels = DynamicImage::ImageRgba16(out);
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
        // 8-bit input is expanded to the full 16-bit range (v * 257).
        assert_eq!(parse_color("#f00").unwrap(), [0xFFFF, 0, 0, 0xFFFF]);
        assert_eq!(parse_color("#f008").unwrap(), [0xFFFF, 0, 0, 0x8888]);
        assert_eq!(parse_color("#ff0000").unwrap(), [0xFFFF, 0, 0, 0xFFFF]);
        assert_eq!(parse_color("#ff000080").unwrap(), [0xFFFF, 0, 0, 0x8080]);
    }

    #[test]
    fn parse_color_16bit_forms() {
        assert_eq!(parse_color("#123456789abc").unwrap(), [0x1234, 0x5678, 0x9ABC, 0xFFFF]);
        assert_eq!(parse_color("#0000ffff00007fff").unwrap(), [0x0000, 0xFFFF, 0x0000, 0x7FFF]);
        // Full scale is full scale in every notation.
        assert_eq!(parse_color("#fff").unwrap(), parse_color("#ffffffffffff").unwrap());
        assert_eq!(parse_color("#ffffff").unwrap(), parse_color("#ffffffffffff").unwrap());
    }

    #[test]
    fn parse_color_bad() {
        assert!(parse_color("ff0000").is_err()); // missing '#'
        assert!(parse_color("#zzz").is_err());
        assert!(parse_color("#12345").is_err()); // length 5 invalid
        assert!(parse_color("#1234567890").is_err()); // length 10 invalid
        assert!(parse_color("#123456789abcdef").is_err()); // length 15 invalid
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
        assert_eq!(c.spec, CanvasSpec::Solid([0x3333, 0x6666, 0x9999, 0xFFFF]));
    }

    #[test]
    fn parse_arg_solid_without_size() {
        // `size:` is optional -- canvas inherits the current image's dimensions
        let c = CanvasConfig::parse_arg("solid,#336699").unwrap();
        assert_eq!(c.size, None);
        assert_eq!(c.spec, CanvasSpec::Solid([0x3333, 0x6666, 0x9999, 0xFFFF]));
    }

    #[test]
    fn parse_arg_linear_fractional_angle() {
        let c = CanvasConfig::parse_arg("size:64x64,linear,16.514,#000000,#ffffff").unwrap();
        assert_eq!(c.size, Some((64, 64)));
        match c.spec {
            CanvasSpec::Linear { angle_deg, stops, .. } => {
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
            CanvasSpec::Radial { center_x, center_y, stops, .. } => {
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
            CanvasSpec::Radial { center_x, center_y, stops, .. } => {
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
            CanvasSpec::Radial { center_x, center_y, stops, .. } => {
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
            GradientStop { pos: 0.0, color: [65535, 0, 0, 65535] },
            GradientStop { pos: 1.0, color: [0, 0, 65535, 65535] },
        ]);
        let lo = pg.eval(0.0);
        let hi = pg.eval(1.0);
        // endpoints should round-trip through Oklab very close to original sRGB
        // (same 240/15-of-255 tolerance as before, scaled to 16 bits)
        assert!(lo[0] > 61_000 && lo[1] < 4_000 && lo[2] < 4_000);
        assert!(hi[0] < 4_000 && hi[1] < 4_000 && hi[2] > 61_000);
        // alpha preserved exactly
        assert_eq!(lo[3], 65535);
        assert_eq!(hi[3], 65535);
    }

    #[test]
    fn gradient_resolves_below_8bit_steps() {
        // The point of the 16-bit pipeline: intermediate samples must be able
        // to land off the 257-value lattice that an expanded 8-bit ramp is
        // stuck on. If this fails we are still quantizing to 8 bits somewhere.
        let pg = PreparedGradient::new(&[
            GradientStop { pos: 0.0, color: [0, 0, 0, 65535] },
            GradientStop { pos: 1.0, color: [65535, 65535, 65535, 65535] },
        ]);
        let off_lattice = (0..=1000).any(|i| pg.eval(i as f64 / 1000.0)[0] % 257 != 0);
        assert!(off_lattice, "gradient output never left the 8-bit lattice");
    }

    #[test]
    fn parse_arg_coons_minimal() {
        // Bare `coons` -- random shape AND random colors from an auto seed.
        let c = CanvasConfig::parse_arg("size:64x64,coons").unwrap();
        assert_eq!(c.size, Some((64, 64)));
        match c.spec {
            CanvasSpec::Coons { seed, colors, easing, transparency } => {
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
            CanvasSpec::Coons { seed, colors, transparency, .. } => {
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
    fn parse_arg_voronoi_defaults_and_styles() {
        let c = CanvasConfig::parse_arg("size:64x64,voronoi").unwrap();
        match c.spec {
            CanvasSpec::Voronoi { seed, cells, style, colors, .. } => {
                assert_eq!(seed, None);
                assert_eq!(cells, 12);
                assert_eq!(style, VoronoiStyle::Sharp);
                assert!(colors.is_none());
            }
            _ => panic!("expected voronoi"),
        }

        // `sharp` and `blob` select the edge treatment; order is free.
        let blob = CanvasConfig::parse_arg("voronoi,blob,cells:30,seed:7").unwrap();
        let blob2 = CanvasConfig::parse_arg("voronoi,seed:7,cells:30,blob").unwrap();
        assert_eq!(blob.spec, blob2.spec);
        match blob.spec {
            CanvasSpec::Voronoi { style, cells, seed, .. } => {
                assert_eq!(style, VoronoiStyle::Blob);
                assert_eq!(cells, 30);
                assert_eq!(seed, Some(7));
            }
            _ => panic!("expected voronoi"),
        }
    }

    #[test]
    fn parse_arg_voronoi_bad_options() {
        // Out of range, non-numeric, unknown key, repeated key.
        assert!(CanvasConfig::parse_arg("voronoi,cells:0").is_err());
        assert!(CanvasConfig::parse_arg("voronoi,cells:abc").is_err());
        assert!(CanvasConfig::parse_arg("voronoi,cels:8").is_err());
        assert!(CanvasConfig::parse_arg("voronoi,cells:8,cells:9").is_err());
        assert!(CanvasConfig::parse_arg("voronoi,softness:0.0").is_err());
        // A single color is not enough to build a palette from.
        assert!(CanvasConfig::parse_arg("voronoi,#ff0000").is_err());
    }

    #[test]
    fn parse_arg_fbm_options_and_stops() {
        let c = CanvasConfig::parse_arg(
            "size:32x32,fbm,seed:3,octaves:4,persistence:0.6,zoom:8,seamless,#000000,#ffffff",
        )
        .unwrap();
        match c.spec {
            CanvasSpec::Fbm { seed, octaves, persistence, zoom, seamless, stops, .. } => {
                assert_eq!(seed, Some(3));
                assert_eq!(octaves, 4);
                assert!((persistence - 0.6).abs() < 1e-12);
                assert!((zoom - 8.0).abs() < 1e-12);
                assert!(seamless);
                assert_eq!(stops.unwrap().len(), 2);
            }
            _ => panic!("expected fbm"),
        }
    }

    #[test]
    fn parse_arg_fbm_positional_stops_are_not_options() {
        // `0.5:#00ff00` contains a colon but is a gradient stop, not `key:value`.
        let c = CanvasConfig::parse_arg("fbm,0:#000000,0.5:#00ff00,1:#ffffff").unwrap();
        match c.spec {
            CanvasSpec::Fbm { stops, .. } => {
                let stops = stops.expect("stops parsed");
                assert_eq!(stops.len(), 3);
                assert!((stops[1].pos - 0.5).abs() < 1e-12);
            }
            _ => panic!("expected fbm"),
        }
    }

    #[test]
    fn parse_arg_fbm_bad_options() {
        assert!(CanvasConfig::parse_arg("fbm,octaves:0").is_err());
        assert!(CanvasConfig::parse_arg("fbm,octaves:99").is_err());
        assert!(CanvasConfig::parse_arg("fbm,persistence:2.0").is_err());
        assert!(CanvasConfig::parse_arg("fbm,seamles").is_err()); // typo'd flag
    }

    #[test]
    fn parse_arg_flame_continuous_flag() {
        let c = CanvasConfig::parse_arg("flame,continuous,seed:5").unwrap();
        match c.spec {
            CanvasSpec::Flame { continuous, .. } => assert!(continuous),
            _ => panic!("expected flame"),
        }
        let d = CanvasConfig::parse_arg("flame,seed:5").unwrap();
        match d.spec {
            CanvasSpec::Flame { continuous, .. } => assert!(!continuous),
            _ => panic!("expected flame"),
        }
    }

    #[test]
    fn continuous_variations_exclude_the_discontinuous_ones() {
        // Exactly polar, heart and disc jump across the branch cut; the rest
        // only ever use theta inside sin/cos, where the jump cancels.
        for v in CONTINUOUS_VARIATIONS {
            assert!(
                !matches!(v, 5 | 7 | 8),
                "variation {} jumps across the branch cut and cannot be cut-free",
                v
            );
        }
        // ...and every other variation is present, so the pool stays varied.
        for v in 0..=16_u8 {
            assert_eq!(
                CONTINUOUS_VARIATIONS.contains(&v),
                !matches!(v, 5 | 7 | 8),
                "variation {} is on the wrong side of the cut-free set",
                v
            );
        }
        // No duplicates, so the modulo draw is uniform over distinct choices.
        let mut sorted = CONTINUOUS_VARIATIONS;
        sorted.sort_unstable();
        let mut deduped = sorted.to_vec();
        deduped.dedup();
        assert_eq!(deduped.len(), CONTINUOUS_VARIATIONS.len());
    }

    #[test]
    fn parse_arg_flow_and_lowpoly_and_flame() {
        let flow = CanvasConfig::parse_arg("flow,seed:1,strands:500,steps:120,width:2").unwrap();
        match flow.spec {
            CanvasSpec::Flow { strands, steps, width, .. } => {
                assert_eq!(strands, 500);
                assert_eq!(steps, 120);
                assert!((width - 2.0).abs() < 1e-12);
            }
            _ => panic!("expected flow"),
        }

        let lp = CanvasConfig::parse_arg("lowpoly,points:40,smooth,seed:9").unwrap();
        match lp.spec {
            CanvasSpec::LowPoly { points, smooth, seed, .. } => {
                assert_eq!(points, 40);
                assert!(smooth);
                assert_eq!(seed, Some(9));
            }
            _ => panic!("expected lowpoly"),
        }

        let fl = CanvasConfig::parse_arg("flame,quality:5,transforms:4,gamma:1.8").unwrap();
        match fl.spec {
            CanvasSpec::Flame { quality, transforms, gamma, .. } => {
                assert_eq!(quality, 5);
                assert_eq!(transforms, 4);
                assert!((gamma - 1.8).abs() < 1e-12);
            }
            _ => panic!("expected flame"),
        }

        // Range checks on the generators that take counts.
        assert!(CanvasConfig::parse_arg("flow,alpha:1.5").is_err());
        assert!(CanvasConfig::parse_arg("lowpoly,points:2").is_err());
        assert!(CanvasConfig::parse_arg("flame,transforms:1").is_err());
        assert!(CanvasConfig::parse_arg("flame,transforms:99").is_err());
    }

    #[test]
    fn spec_tokens_splits_options_flags_and_colors() {
        let t = SpecTokens::split(&["seed:42", "blob", "#ff0000", "0.5:#00ff00"]);
        assert_eq!(t.opts, vec![("seed", "42")]);
        assert_eq!(t.flags, vec!["blob"]);
        assert_eq!(t.colors, vec!["#ff0000", "0.5:#00ff00"]);
    }

    #[test]
    fn spec_tokens_take_leaves_duplicates_for_finish() {
        let mut t = SpecTokens::split(&["cells:8", "cells:9"]);
        assert_eq!(t.take("cells"), Some("8"));
        // The repeat is still there, so `finish` is what rejects it.
        assert!(t.finish("err").is_err());
    }

    #[test]
    fn random_palette_is_deterministic_and_in_gamut() {
        let mut r1 = SplitMix64::new(1234);
        let mut r2 = SplitMix64::new(1234);
        let p1 = random_palette(6, &mut r1);
        let p2 = random_palette(6, &mut r2);
        assert_eq!(p1.len(), 6);
        assert_eq!(p1, p2);
        for (lab, a) in p1 {
            assert!((0.0..=1.0).contains(&lab.l), "lightness {} out of range", lab.l);
            assert!(lab.a.abs() < 0.5 && lab.b.abs() < 0.5);
            assert_eq!(a, MAX16);
        }
    }

    #[test]
    fn palette_sample_hits_endpoints_and_midpoint() {
        let pal = vec![
            (Oklab { l: 0.0, a: 0.0, b: 0.0 }, MAX16),
            (Oklab { l: 1.0, a: 0.0, b: 0.0 }, MAX16),
        ];
        assert!((palette_sample(&pal, 0.0).0.l - 0.0).abs() < 1e-6);
        assert!((palette_sample(&pal, 1.0).0.l - 1.0).abs() < 1e-6);
        assert!((palette_sample(&pal, 0.5).0.l - 0.5).abs() < 1e-6);
        // Out-of-range input clamps rather than panicking on the index.
        assert!((palette_sample(&pal, -3.0).0.l - 0.0).abs() < 1e-6);
        assert!((palette_sample(&pal, 7.0).0.l - 1.0).abs() < 1e-6);
    }

    #[test]
    fn hash2d_is_stable_and_well_spread() {
        // Same inputs -> same hash, neighbours -> different hashes.
        assert_eq!(hash2d(3, -7, 42), hash2d(3, -7, 42));
        assert_ne!(hash2d(3, -7, 42), hash2d(3, -7, 43));
        assert_ne!(hash2d(3, -7, 42), hash2d(4, -7, 42));
        assert_ne!(hash2d(3, -7, 42), hash2d(3, -6, 42));
        // Jitter stays inside the cell, which is what keeps the 3x3 search valid.
        for i in 0..500_i64 {
            let (jx, jy) = hash_unit2(hash2d(i, i * 7 - 3, 99));
            assert!((0.0..1.0).contains(&jx));
            assert!((0.0..1.0).contains(&jy));
        }
    }

    #[test]
    fn flame_variations_stay_finite() {
        // Every variation is hit with awkward inputs, including the origin
        // where several of them divide by r.
        for v in 0..=16_u8 {
            for &(x, y) in &[(0.0, 0.0), (1e-9, -1e-9), (0.7, -0.3), (-2.5, 4.25), (1e3, -1e3)] {
                let (ox, oy) = flame_variation(v, x, y);
                assert!(
                    ox.is_finite() && oy.is_finite(),
                    "variation {} produced a non-finite result at ({}, {})",
                    v,
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
