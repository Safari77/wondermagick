use crate::{arg_parse_err::ArgParseErr, error::MagickError, image::Image, wm_err};
use cosmic_text::{Align, Attrs, Buffer, Family, FontSystem, Metrics, Shaping, SwashCache};
use image::{DynamicImage, RgbaImage};
use qrcode::{EcLevel, QrCode};
use rayon::prelude::*;
use tiny_skia::{BlendMode, ColorU8, Pixmap, PixmapPaint, PremultipliedColorU8, Transform};

/// A parsed QR code block extracted from the text field.
#[derive(Debug, Clone, PartialEq)]
struct QrBlock {
    pub ec_level: EcLevel,
    pub content: String,
    /// Gaussian blur sigma for the region under the QR code (0.0 = disabled).
    pub blur_sigma: f32,
    /// Light (background) color of the QR code modules.
    pub light_color: (u8, u8, u8, u8),
}

/// A segment of the text field: either plain text or a QR code block.
#[derive(Debug, Clone, PartialEq)]
enum TextSegment {
    Plain(String),
    Qr(QrBlock),
}

/// Parse the text field into segments, handling `{QR:EcLevel:blur:light_color:content}` blocks
/// and `\{QR` escape sequences.
///
/// - `{QR:L:0:#00000000:hello}` → QR block, no blur, transparent light color
/// - `{QR:H:3.0:#FFFFFFCC:https://x.com}` → QR block, blur sigma=3.0, semi-opaque white light
/// - `\{QR:L:...}` → literal text "{QR:L:...}"
fn parse_text_segments(text: &str) -> Result<Vec<TextSegment>, ArgParseErr> {
    let mut segments: Vec<TextSegment> = Vec::new();
    let mut plain = String::new();
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        // Handle escape: \{QR → literal {QR
        if chars[i] == '\\' && i + 1 < len && chars[i + 1] == '{' {
            // Check if this looks like an escaped QR block
            let rest: String = chars[i + 1..].iter().collect();
            if rest.starts_with("{QR") {
                plain.push('{');
                i += 2; // skip \ and {
                continue;
            }
            // Not a QR escape, keep backslash as-is
            plain.push(chars[i]);
            i += 1;
            continue;
        }

        // Detect {QR:...:...:...:...}
        if chars[i] == '{' {
            let rest: String = chars[i..].iter().collect();
            if rest.starts_with("{QR:") {
                // Find the closing brace
                if let Some(close_pos) = rest.find('}') {
                    let inner = &rest[4..close_pos]; // after "{QR:" and before "}"

                    // Parse: EcLevel:blur_sigma:light_color:content
                    // Split at first 3 colons to get 4 parts
                    let mut parts = inner.splitn(4, ':');
                    let ec_str = parts
                        .next()
                        .ok_or_else(|| ArgParseErr::with_msg("QR block missing EcLevel"))?;
                    let blur_str = parts
                        .next()
                        .ok_or_else(|| ArgParseErr::with_msg("QR block missing blur sigma"))?;
                    let light_str = parts
                        .next()
                        .ok_or_else(|| ArgParseErr::with_msg("QR block missing light_color"))?;
                    let content = parts
                        .next()
                        .ok_or_else(|| ArgParseErr::with_msg("QR block missing content"))?;

                    let ec_level = match ec_str {
                        "L" => EcLevel::L,
                        "M" => EcLevel::M,
                        "Q" => EcLevel::Q,
                        "H" => EcLevel::H,
                        _ => {
                            return Err(ArgParseErr::with_msg("QR EcLevel must be L, M, Q, or H"));
                        }
                    };

                    let blur_sigma = blur_str.parse::<f32>().map_err(|_| {
                        ArgParseErr::with_msg("QR blur must be a number (0 to disable, e.g. 3.0)")
                    })?;

                    let light_color = parse_hex_color(light_str)?;

                    // Flush accumulated plain text
                    if !plain.is_empty() {
                        segments.push(TextSegment::Plain(std::mem::take(&mut plain)));
                    }

                    segments.push(TextSegment::Qr(QrBlock {
                        ec_level,
                        content: content.to_string(),
                        blur_sigma,
                        light_color,
                    }));

                    i += close_pos + 1; // skip past '}'
                    continue;
                }
            }
        }

        plain.push(chars[i]);
        i += 1;
    }

    if !plain.is_empty() {
        segments.push(TextSegment::Plain(plain));
    }

    Ok(segments)
}

#[derive(Debug, Clone, PartialEq)]
pub enum FontSize {
    Absolute(f32),
    RelativePercent(f32),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Position {
    Center,
    Absolute(f32),
    Percent(f32),
    Em(f32),
}

impl Position {
    fn parse(s: &str) -> Result<Self, ArgParseErr> {
        let s = s.trim().to_lowercase();
        if s == "center" || s == "middle" {
            Ok(Position::Center)
        } else if let Some(pct) = s.strip_suffix('%') {
            Ok(Position::Percent(pct.parse::<f32>().map_err(|_| {
                ArgParseErr::with_msg("invalid percentage position")
            })?))
        } else if let Some(em) = s.strip_suffix("em") {
            Ok(Position::Em(em.parse::<f32>().map_err(|_| {
                ArgParseErr::with_msg("invalid em position")
            })?))
        } else {
            Ok(Position::Absolute(s.parse::<f32>().map_err(|_| {
                ArgParseErr::with_msg("invalid absolute position")
            })?))
        }
    }

    /// Resolve position so that 0% = start-aligned, 100% = end-aligned, 50% = centered.
    fn resolve(&self, container_size: f32, item_size: f32, font_size: f32) -> f32 {
        match self {
            Position::Center => (container_size - item_size) / 2.0,
            Position::Absolute(v) => *v,
            Position::Percent(pct) => (container_size - item_size) * (pct / 100.0),
            Position::Em(v) => v * font_size,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TextEffect {
    /// No background effect, raw text.
    None,
    /// Gaussian-blur the region behind the text bounding box.
    Blur { sigma: f32 },
    /// Like Blur, but the edges of the blurred region graduate smoothly
    /// into the unblurred background instead of cutting off sharply.
    GradualBlur { sigma: f32 },
    /// Subtitle-style outline via morphological dilation of the text alpha mask.
    Outline {
        thickness: u32,
        color: (u8, u8, u8, u8),
    },
    /// Soft drop shadow behind the text glyphs, offset by (dx, dy) pixels
    /// and gaussian-blurred to give depth.
    Shadow {
        dx: f32,
        dy: f32,
        sigma: f32,
        color: (u8, u8, u8, u8),
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct TextConfig {
    pub effect: TextEffect,
    pub text: String,
    pub font_name: String,
    pub font_size: FontSize,
    pub color: (u8, u8, u8, u8), // (R, G, B, A)
    pub rotation: f32,
    pub justify: Align,
    pub x: Position,
    pub y: Position,
}

fn parse_hex_color(hex: &str) -> Result<(u8, u8, u8, u8), ArgParseErr> {
    let hex = hex.trim().trim_start_matches('#');
    if hex.len() == 6 || hex.len() == 8 {
        let r =
            u8::from_str_radix(&hex[0..2], 16).map_err(|_| ArgParseErr::with_msg("invalid red"))?;
        let g = u8::from_str_radix(&hex[2..4], 16)
            .map_err(|_| ArgParseErr::with_msg("invalid green"))?;
        let b = u8::from_str_radix(&hex[4..6], 16)
            .map_err(|_| ArgParseErr::with_msg("invalid blue"))?;
        let a = if hex.len() == 8 {
            u8::from_str_radix(&hex[6..8], 16)
                .map_err(|_| ArgParseErr::with_msg("invalid alpha"))?
        } else {
            255
        };
        Ok((r, g, b, a))
    } else {
        Err(ArgParseErr::with_msg("color must be #RRGGBB or #RRGGBBAA"))
    }
}

impl TextConfig {
    /// Format: "effect,text,font_name,font_size,color,rotation,justify,x,y"
    ///
    /// Effect values:
    ///   - `none`                 — plain text, no background effect
    ///   - `blur:5.0`             — blur background behind text (sigma=5.0)
    ///   - `gradualblur:5.0`      — blur with smooth graduated edges (sigma=5.0)
    ///   - `outline:3:#000000FF`  — subtitle-style outline (thickness 3, black)
    ///   - `shadow:3:3:4.0:#00000080` — drop shadow (dx, dy, sigma, color)
    ///
    /// Example: "outline:3:#000000FF,Hello\\nWorld,Arial,5%,#FFFFFF,-45.0,center,center,80%"
    /// Position units: px (absolute), % (0%=start, 100%=end-aligned), em (font-size-relative), center
    pub fn parse_arg(s: &str) -> Result<Self, ArgParseErr> {
        // Split effect (first field) from the rest.
        // Effect uses colons internally, never commas, so the first comma is the boundary.
        let first_comma = s.find(',').ok_or_else(|| {
            ArgParseErr::with_msg(
                "text requires 9 comma-separated values: \
                 effect,text,font_name,font_size,color,rotation,justify,x,y",
            )
        })?;
        let effect_str = &s[..first_comma];
        let rest = &s[first_comma + 1..];

        let effect = Self::parse_effect(effect_str)?;

        let mut parts: Vec<&str> = rest.rsplitn(8, ',').collect();
        if parts.len() != 8 {
            return Err(ArgParseErr::with_msg(
                "text requires 9 comma-separated values: \
                 effect,text,font_name,font_size,color,rotation,justify,x,y",
            ));
        }
        parts.reverse();

        let text = parts[0].replace("\\n", "\n");
        let font_name = parts[1].trim().to_string();

        let size_str = parts[2].trim();
        let font_size = if let Some(pct) = size_str.strip_suffix('%') {
            FontSize::RelativePercent(
                pct.parse::<f32>()
                    .map_err(|_| ArgParseErr::with_msg("invalid pct size"))?,
            )
        } else {
            FontSize::Absolute(
                size_str
                    .parse::<f32>()
                    .map_err(|_| ArgParseErr::with_msg("invalid abs size"))?,
            )
        };

        let color = parse_hex_color(parts[3])?;

        let rotation = parts[4]
            .trim()
            .parse::<f32>()
            .map_err(|_| ArgParseErr::with_msg("invalid rotation (must be float degrees)"))?;

        let justify = match parts[5].trim().to_lowercase().as_str() {
            "left" => Align::Left,
            "center" | "middle" => Align::Center,
            "right" => Align::Right,
            _ => {
                return Err(ArgParseErr::with_msg(
                    "justify must be left, middle, or right",
                ))
            }
        };

        let x = Position::parse(parts[6])?;
        let y = Position::parse(parts[7])?;

        Ok(Self {
            effect,
            text,
            font_name,
            font_size,
            color,
            rotation,
            justify,
            x,
            y,
        })
    }

    fn parse_effect(s: &str) -> Result<TextEffect, ArgParseErr> {
        let s = s.trim();
        if s.eq_ignore_ascii_case("none") {
            return Ok(TextEffect::None);
        }
        if let Some(sigma_str) = s.strip_prefix("blur:") {
            let sigma = sigma_str.parse::<f32>().map_err(|_| {
                ArgParseErr::with_msg("blur effect requires a float sigma (e.g. 'blur:5.0')")
            })?;
            return Ok(TextEffect::Blur { sigma });
        }
        if let Some(sigma_str) = s.strip_prefix("gradualblur:") {
            let sigma = sigma_str.parse::<f32>().map_err(|_| {
                ArgParseErr::with_msg(
                    "gradualblur effect requires a float sigma (e.g. 'gradualblur:5.0')",
                )
            })?;
            return Ok(TextEffect::GradualBlur { sigma });
        }
        if let Some(rest) = s.strip_prefix("outline:") {
            // Format: outline:thickness:#color
            let colon_pos = rest.find(':').ok_or_else(|| {
                ArgParseErr::with_msg(
                    "outline effect requires thickness:color (e.g. 'outline:3:#000000FF')",
                )
            })?;
            let thickness_str = &rest[..colon_pos];
            let color_str = &rest[colon_pos + 1..];
            let thickness = thickness_str.parse::<u32>().map_err(|_| {
                ArgParseErr::with_msg("outline thickness must be a positive integer")
            })?;
            if thickness == 0 {
                return Err(ArgParseErr::with_msg("outline thickness must be >= 1"));
            }
            let color = parse_hex_color(color_str)?;
            return Ok(TextEffect::Outline { thickness, color });
        }
        if let Some(rest) = s.strip_prefix("shadow:") {
            // Format: shadow:dx:dy:sigma:#color
            let mut parts = rest.splitn(4, ':');
            let dx_str = parts
                .next()
                .ok_or_else(|| ArgParseErr::with_msg("shadow effect missing dx"))?;
            let dy_str = parts
                .next()
                .ok_or_else(|| ArgParseErr::with_msg("shadow effect missing dy"))?;
            let sigma_str = parts
                .next()
                .ok_or_else(|| ArgParseErr::with_msg("shadow effect missing sigma"))?;
            let color_str = parts
                .next()
                .ok_or_else(|| ArgParseErr::with_msg("shadow effect missing color"))?;
            let dx = dx_str.parse::<f32>().map_err(|_| {
                ArgParseErr::with_msg("shadow dx must be a float (e.g. 'shadow:3:3:4.0:#00000080')")
            })?;
            let dy = dy_str.parse::<f32>().map_err(|_| {
                ArgParseErr::with_msg("shadow dy must be a float (e.g. 'shadow:3:3:4.0:#00000080')")
            })?;
            let sigma = sigma_str.parse::<f32>().map_err(|_| {
                ArgParseErr::with_msg(
                    "shadow sigma must be a float (e.g. 'shadow:3:3:4.0:#00000080')",
                )
            })?;
            let color = parse_hex_color(color_str)?;
            return Ok(TextEffect::Shadow {
                dx,
                dy,
                sigma,
                color,
            });
        }
        Err(ArgParseErr::with_msg(
            "effect must be 'none', 'blur:<sigma>', 'gradualblur:<sigma>', \
             'outline:<thickness>:<#color>', or 'shadow:<dx>:<dy>:<sigma>:<#color>'",
        ))
    }
}

/// Blur the region of `main_pixmap` that falls under a rotated mask.
///
/// `mask_src` is a solid-filled pixmap (e.g. the size of the QR or text layer).
/// `transform` is the same transform used to composite the content.
/// The blurred source comes from `source_image`.
fn apply_blur_under_rotated_region(
    main_pixmap: &mut Pixmap,
    source_image: &Image,
    mask_src: &Pixmap,
    transform: Transform,
    sigma: f32,
    img_w: u32,
    img_h: u32,
) {
    let blurred = source_image.pixels.blur(sigma);
    let blurred_rgba = blurred.to_rgba8();

    // Draw the solid mask through the rotation transform to get the rotated footprint
    let mut mask_full = Pixmap::new(img_w, img_h).expect("Failed to allocate mask pixmap");
    mask_full.draw_pixmap(
        0,
        0,
        mask_src.as_ref(),
        &PixmapPaint {
            opacity: 1.0,
            blend_mode: BlendMode::SourceOver,
            quality: tiny_skia::FilterQuality::Bilinear,
        },
        transform,
        None,
    );

    // Blend: where mask alpha > 0, lerp blurred into main
    for (i, mask_px) in mask_full.pixels().iter().enumerate() {
        let ma = mask_px.alpha() as u32;
        if ma == 0 {
            continue;
        }
        let x = (i as u32) % img_w;
        let y = (i as u32) / img_w;
        let bp = blurred_rgba.get_pixel(x, y);
        let blurred_pre = ColorU8::from_rgba(bp[0], bp[1], bp[2], bp[3]).premultiply();

        let orig = main_pixmap.pixels()[i];
        let inv = 255 - ma;
        let out_r = ((blurred_pre.red() as u32 * ma + orig.red() as u32 * inv) / 255) as u8;
        let out_g = ((blurred_pre.green() as u32 * ma + orig.green() as u32 * inv) / 255) as u8;
        let out_b = ((blurred_pre.blue() as u32 * ma + orig.blue() as u32 * inv) / 255) as u8;
        let out_a = ((blurred_pre.alpha() as u32 * ma + orig.alpha() as u32 * inv) / 255) as u8;

        if let Some(c) = PremultipliedColorU8::from_rgba(out_r, out_g, out_b, out_a) {
            main_pixmap.pixels_mut()[i] = c;
        }
    }
}

/// Morphological dilation on an alpha channel using a circular structuring element.
/// `thickness` is the kernel diameter (will be forced to odd).
fn dilate_alpha(alpha: &[u8], width: u32, height: u32, thickness: u32) -> Vec<u8> {
    let ksize = if thickness.is_multiple_of(2) {
        thickness + 1
    } else {
        thickness
    };
    let half = ksize / 2;
    let half_f = half as f64;
    let r_sq = half_f * half_f;

    let w = width as usize;
    let mut output = vec![0u8; alpha.len()];

    // Process rows in parallel
    output
        .par_chunks_mut(w)
        .enumerate()
        .for_each(|(y_idx, row)| {
            let y = y_idx as u32;
            let y_start = y.saturating_sub(half);
            let y_end = (y + half).min(height - 1);

            for x in 0..width {
                let x_start = x.saturating_sub(half);
                let x_end = (x + half).min(width - 1);
                let mut local_max = 0u8;

                for ky in y_start..=y_end {
                    for kx in x_start..=x_end {
                        // Circular structuring element
                        let dx = kx.abs_diff(x) as f64;
                        let dy = ky.abs_diff(y) as f64;
                        if dx * dx + dy * dy <= r_sq {
                            let val = alpha[ky as usize * w + kx as usize];
                            if val > local_max {
                                local_max = val;
                            }
                        }
                    }
                }
                row[x as usize] = local_max;
            }
        });

    output
}

/// Render a QR code block onto the image, using the same position/rotation/color
/// system as text rendering. The QR code size is determined by font_size.
/// Renders at 4x internal resolution to anti-alias rotated edges.
fn render_qr_block(
    image: &mut Image,
    config: &TextConfig,
    qr: &QrBlock,
) -> Result<(), MagickError> {
    let rgba_img = image.pixels.to_rgba8();
    let (img_w, img_h) = rgba_img.dimensions();

    let qr_size = match config.font_size {
        FontSize::Absolute(v) => v,
        FontSize::RelativePercent(pct) => (img_h as f32 * (pct / 100.0)).max(1.0),
    };

    let code = QrCode::with_error_correction_level(qr.content.as_bytes(), qr.ec_level)
        .map_err(|e| wm_err!("QR generation failed: {}", e))?;

    let (r, g, b, a) = config.color;
    let (lr, lg, lb, la) = qr.light_color;

    // Render QR at 4x supersampled resolution for smooth rotated edges
    let ss = 4u32;
    let ss_size = qr_size as u32 * ss;
    let qr_rgba: RgbaImage = code
        .render::<image::Rgba<u8>>()
        .dark_color(image::Rgba([r, g, b, a]))
        .light_color(image::Rgba([lr, lg, lb, la]))
        .min_dimensions(ss_size, ss_size)
        .build();

    let (ss_w, ss_h) = qr_rgba.dimensions();
    // Final display size after downscale
    let qw = ss_w / ss;
    let qh = ss_h / ss;

    // Build a tiny_skia pixmap from the supersampled QR image
    let mut qr_pixmap = Pixmap::new(ss_w, ss_h).expect("Failed to allocate QR pixmap");
    for (src, dst) in qr_rgba.pixels().zip(qr_pixmap.pixels_mut()) {
        *dst = ColorU8::from_rgba(src[0], src[1], src[2], src[3]).premultiply();
    }

    // Compute the bounding box of the QR after rotation so positioning
    // accounts for the full rotated extent (prevents corner clipping).
    let angle_rad = config.rotation.to_radians();
    let cos_a = angle_rad.cos().abs();
    let sin_a = angle_rad.sin().abs();
    let rot_w = qw as f32 * cos_a + qh as f32 * sin_a;
    let rot_h = qw as f32 * sin_a + qh as f32 * cos_a;

    // Resolve position using the *rotated* bounding box size so that e.g.
    // x=0 means the leftmost rotated corner sits at x=0.
    let desired_x = config.x.resolve(img_w as f32, rot_w, qr_size);
    let desired_y = config.y.resolve(img_h as f32, rot_h, qr_size);

    // The rotation pivot is at (qw/2, qh/2) relative to the translate origin.
    // After rotation the bounding box top-left shifts by (rot_w-qw)/2 in each axis.
    // Compensate so the *rotated* box lands at (desired_x, desired_y).
    let start_x = desired_x + (rot_w - qw as f32) / 2.0;
    let start_y = desired_y + (rot_h - qh as f32) / 2.0;

    // Scale down from supersampled size + translate + rotate
    let scale = 1.0 / ss as f32;
    let transform = Transform::from_translate(start_x, start_y)
        .pre_concat(Transform::from_rotate_at(
            config.rotation,
            qw as f32 / 2.0,
            qh as f32 / 2.0,
        ))
        .pre_concat(Transform::from_scale(scale, scale));

    // Start with the original image as the base
    let mut main_pixmap = Pixmap::new(img_w, img_h).expect("Failed to allocate main pixmap");
    for (src, dst) in rgba_img.pixels().zip(main_pixmap.pixels_mut()) {
        *dst = ColorU8::from_rgba(src[0], src[1], src[2], src[3]).premultiply();
    }

    // If blur_sigma > 0, blur the region under the *rotated* QR footprint.
    if qr.blur_sigma > 0.0 {
        let mut mask_src = Pixmap::new(ss_w, ss_h).expect("Failed to allocate mask source pixmap");
        for px in mask_src.pixels_mut() {
            *px = PremultipliedColorU8::from_rgba(255, 255, 255, 255).unwrap();
        }
        apply_blur_under_rotated_region(
            &mut main_pixmap,
            image,
            &mask_src,
            transform,
            qr.blur_sigma,
            img_w,
            img_h,
        );
    }

    // Composite the QR code on top of the (optionally blurred) base
    main_pixmap.draw_pixmap(
        0,
        0,
        qr_pixmap.as_ref(),
        &PixmapPaint {
            opacity: 1.0,
            blend_mode: BlendMode::SourceOver,
            quality: tiny_skia::FilterQuality::Bilinear,
        },
        transform,
        None,
    );

    let mut out_rgba = RgbaImage::new(img_w, img_h);
    for (src, dst) in main_pixmap.pixels().iter().zip(out_rgba.pixels_mut()) {
        let un_pre = src.demultiply();
        *dst = image::Rgba([un_pre.red(), un_pre.green(), un_pre.blue(), un_pre.alpha()]);
    }

    image.pixels = DynamicImage::ImageRgba8(out_rgba);
    Ok(())
}

pub fn render_text(image: &mut Image, config: &TextConfig) -> Result<(), MagickError> {
    // Parse text for QR blocks and escape sequences
    let segments = parse_text_segments(&config.text).map_err(|e| wm_err!("{:?}", e.message))?;

    // If the entire text is a single QR block, render only the QR code
    if segments.len() == 1 {
        if let TextSegment::Qr(ref qr) = segments[0] {
            return render_qr_block(image, config, qr);
        }
    }

    // For mixed content or plain text, render QR blocks first (each one composited),
    // then render the remaining plain text on top.
    for seg in &segments {
        if let TextSegment::Qr(ref qr) = seg {
            render_qr_block(image, config, qr)?;
        }
    }

    // Collect plain text portions (skip QR blocks)
    let plain_text: String = segments
        .iter()
        .map(|s| match s {
            TextSegment::Plain(t) => t.as_str(),
            TextSegment::Qr(_) => "",
        })
        .collect();

    // If there's no plain text left, we're done
    if plain_text.trim().is_empty() {
        return Ok(());
    }

    // Render the plain text portion using the original text rendering pipeline,
    // but with QR blocks stripped out.
    let text_config = TextConfig {
        text: plain_text,
        ..config.clone()
    };
    render_text_inner(image, &text_config)
}

fn render_text_inner(image: &mut Image, config: &TextConfig) -> Result<(), MagickError> {
    let rgba_img = image.pixels.to_rgba8();
    let (img_w, img_h) = rgba_img.dimensions();

    let mut font_system = FontSystem::new();
    let mut swash_cache = SwashCache::new();

    // Support colon-separated font names as fallback list (e.g. "Iosevka:Twemoji:sans-serif")
    let font_names: Vec<&str> = config.font_name.split(':').map(|s| s.trim()).collect();
    let mut available_fonts: Vec<&str> = Vec::new();
    for name in &font_names {
        let found = font_system.db().faces().any(|info| {
            info.families
                .iter()
                .any(|(fname, _): &(String, _)| fname.eq_ignore_ascii_case(name))
        });
        if found {
            available_fonts.push(name);
        } else {
            eprintln!("warning: font '{}' not found on this system", name);
        }
    }
    if available_fonts.is_empty() {
        eprintln!("warning: none of the specified fonts were found, text may be invisible");
        available_fonts.push(font_names[0]);
    }
    let primary_font = available_fonts[0];

    let initial_font_size = match config.font_size {
        FontSize::Absolute(v) => v,
        FontSize::RelativePercent(pct) => (img_h as f32 * (pct / 100.0)).max(1.0),
    };

    // Autofit: measure natural text size, shrink font if it exceeds image bounds
    let mut font_size = initial_font_size;
    let (actual_w, actual_h, line_height) = loop {
        let lh = font_size * 1.2;
        let metrics = Metrics::new(font_size, lh);
        let mut buf = Buffer::new(&mut font_system, metrics);
        {
            let mut br = buf.borrow_with(&mut font_system);
            let attrs = Attrs::new().family(Family::Name(primary_font));
            // Measure with left-align and unconstrained width to get natural text extent
            br.set_text(&config.text, &attrs, Shaping::Advanced, Some(Align::Left));
            br.set_size(None, None);
            br.shape_until_scroll(true);

            let mut tw: f32 = 0.0;
            let mut th: f32 = 0.0;
            for run in br.layout_runs() {
                tw = tw.max(run.line_w);
                th = th.max(run.line_y + lh);
            }

            if (tw <= img_w as f32 && th <= img_h as f32) || font_size <= 1.0 {
                break (tw, th, lh);
            }

            let scale = (img_w as f32 / tw).min(img_h as f32 / th);
            let new_size = (font_size * scale).floor().max(1.0);
            if new_size >= font_size {
                break (tw, th, lh);
            }
            font_size = new_size;
        }
    };

    // Padding to avoid clipping ascenders/descenders/italics.
    // Position resolve uses actual_w/actual_h (visible text bounds),
    // then we subtract pad to account for the pixmap's extra margin.
    let pad = font_size;
    let text_w = actual_w + pad * 2.0;
    let text_h = actual_h + pad * 2.0;

    // Final layout with correct justification relative to the actual text width
    let metrics = Metrics::new(font_size, line_height);
    let mut buffer = Buffer::new(&mut font_system, metrics);
    {
        let mut buffer_ref = buffer.borrow_with(&mut font_system);
        let attrs = Attrs::new().family(Family::Name(primary_font));
        buffer_ref.set_text(
            &config.text,
            &attrs,
            Shaping::Advanced,
            Some(config.justify),
        );
        // Use actual text width so justification (center/right) works correctly
        buffer_ref.set_size(Some(actual_w), None);
        buffer_ref.shape_until_scroll(true);
    }

    let tw_u32 = (text_w.ceil() as u32).max(1);
    let th_u32 = (text_h.ceil() as u32).max(1);

    let mut text_pixmap = Pixmap::new(tw_u32, th_u32).expect("Failed to allocate text pixmap");

    let mut total_glyphs = 0u32;

    // Track glyphs that failed to rasterize for fallback rendering
    struct FailedGlyph {
        text: String,
        x: f32,
        y: f32,
    }
    let mut failed: Vec<FailedGlyph> = Vec::new();

    for run in buffer.layout_runs() {
        for glyph in run.glyphs.iter() {
            total_glyphs += 1;
            let physical_glyph = glyph.physical((0., 0.), 1.0);
            if let Some(img) = swash_cache.get_image(&mut font_system, physical_glyph.cache_key) {
                // Offset by pad to utilize the padding margin and avoid clipping.
                // run.line_y positions each line vertically for multiline text.
                let gx = physical_glyph.x as f32 + img.placement.left as f32 + pad;
                let gy = run.line_y + physical_glyph.y as f32 - img.placement.top as f32 + pad;

                draw_glyph_pixels(&mut text_pixmap, tw_u32, th_u32, img, gx, gy, config.color);
            } else {
                // Save position and text for fallback font rendering
                // Snap to char boundaries since glyph byte ranges may land mid-codepoint
                let start = (0..=glyph.start)
                    .rev()
                    .find(|&i| config.text.is_char_boundary(i))
                    .unwrap_or(0);
                let end = (glyph.end..=config.text.len())
                    .find(|&i| config.text.is_char_boundary(i))
                    .unwrap_or(config.text.len());
                let ch = &config.text[start..end];
                failed.push(FailedGlyph {
                    text: ch.to_string(),
                    x: physical_glyph.x as f32,
                    y: run.line_y + physical_glyph.y as f32,
                });
            }
        }
    }

    // Attempt to render failed glyphs with fallback fonts
    if !failed.is_empty() && available_fonts.len() > 1 {
        for &fallback_name in &available_fonts[1..] {
            let mut still_failed = Vec::new();
            for fg in failed {
                let metrics = Metrics::new(font_size, line_height);
                let mut fb_buf = Buffer::new(&mut font_system, metrics);
                {
                    let mut fb_br = fb_buf.borrow_with(&mut font_system);
                    let attrs = Attrs::new().family(Family::Name(fallback_name));
                    fb_br.set_text(&fg.text, &attrs, Shaping::Advanced, None);
                    fb_br.set_size(None, None);
                    fb_br.shape_until_scroll(true);
                }

                let mut rendered = false;
                for run in fb_buf.layout_runs() {
                    for glyph in run.glyphs.iter() {
                        let pg = glyph.physical((0., 0.), 1.0);
                        if let Some(img) = swash_cache.get_image(&mut font_system, pg.cache_key) {
                            rendered = true;
                            // Draw at the original position from primary font layout
                            let gx = fg.x + img.placement.left as f32 + pad;
                            let gy = fg.y - img.placement.top as f32 + pad;
                            draw_glyph_pixels(
                                &mut text_pixmap,
                                tw_u32,
                                th_u32,
                                img,
                                gx,
                                gy,
                                config.color,
                            );
                        }
                    }
                }
                if !rendered {
                    still_failed.push(fg);
                }
            }
            failed = still_failed;
            if failed.is_empty() {
                break;
            }
        }
    }

    if !failed.is_empty() {
        eprintln!(
            "warning: {}/{} glyphs could not be rendered with any specified font \
             (color bitmap emoji require a monochrome font like Noto Emoji or Twemoji)",
            failed.len(),
            total_glyphs
        );
    }

    let mut main_pixmap = Pixmap::new(img_w, img_h).expect("Failed to allocate main pixmap");
    for (src, dst) in rgba_img.pixels().zip(main_pixmap.pixels_mut()) {
        *dst = ColorU8::from_rgba(src[0], src[1], src[2], src[3]).premultiply();
    }

    // Position based on actual visible text bounds, then subtract pad
    // so the pixmap's padding margin extends outside the positioned area
    let start_x = config.x.resolve(img_w as f32, actual_w, font_size) - pad;
    let start_y = config.y.resolve(img_h as f32, actual_h, font_size) - pad;

    let transform = Transform::from_translate(start_x, start_y).pre_concat(
        Transform::from_rotate_at(config.rotation, text_w / 2.0, text_h / 2.0),
    );

    let paint = PixmapPaint {
        opacity: 1.0,
        blend_mode: BlendMode::SourceOver,
        quality: tiny_skia::FilterQuality::Bilinear,
    };

    // Apply the configured background effect
    match &config.effect {
        TextEffect::None => {}
        TextEffect::Blur { sigma } => {
            // Scan the text pixmap alpha to find the tight bounding box of
            // the actually-rendered glyphs. This avoids line-height padding
            // that makes the blur region asymmetric.
            let pixels = text_pixmap.pixels();
            let mut min_x = tw_u32;
            let mut max_x = 0u32;
            let mut min_y = th_u32;
            let mut max_y = 0u32;
            for py in 0..th_u32 {
                for px in 0..tw_u32 {
                    if pixels[(py * tw_u32 + px) as usize].alpha() > 0 {
                        min_x = min_x.min(px);
                        max_x = max_x.max(px);
                        min_y = min_y.min(py);
                        max_y = max_y.max(py);
                    }
                }
            }

            if max_x >= min_x && max_y >= min_y {
                let glyph_w = (max_x - min_x + 1) as f32;
                let glyph_h = (max_y - min_y + 1) as f32;

                // Small symmetric margin around the tight glyph bounds
                let blur_margin = font_size * 0.3;
                let mask_w = (glyph_w + blur_margin * 2.0).ceil() as u32;
                let mask_h = (glyph_h + blur_margin * 2.0).ceil() as u32;
                let mut mask_src = Pixmap::new(mask_w.max(1), mask_h.max(1))
                    .expect("Failed to allocate blur mask pixmap");
                for px in mask_src.pixels_mut() {
                    *px = PremultipliedColorU8::from_rgba(255, 255, 255, 255).unwrap();
                }

                // The glyph bounding box top-left within the text pixmap is (min_x, min_y).
                // The text pixmap top-left on the canvas is (start_x, start_y).
                // So the glyph top-left on the canvas is (start_x + min_x, start_y + min_y).
                // The mask starts blur_margin before that.
                let mask_x = start_x + min_x as f32 - blur_margin;
                let mask_y = start_y + min_y as f32 - blur_margin;

                // Rotate around the same pivot as the text pixmap so blur stays aligned.
                // Text pivots at (start_x + text_w/2, start_y + text_h/2) in canvas space.
                let pivot_x = start_x + text_w / 2.0 - mask_x;
                let pivot_y = start_y + text_h / 2.0 - mask_y;
                let mask_transform = Transform::from_translate(mask_x, mask_y)
                    .pre_concat(Transform::from_rotate_at(config.rotation, pivot_x, pivot_y));

                apply_blur_under_rotated_region(
                    &mut main_pixmap,
                    image,
                    &mask_src,
                    mask_transform,
                    *sigma,
                    img_w,
                    img_h,
                );
            }
        }
        TextEffect::GradualBlur { sigma } => {
            // Same tight glyph bounding box scan as Blur
            let pixels = text_pixmap.pixels();
            let mut min_x = tw_u32;
            let mut max_x = 0u32;
            let mut min_y = th_u32;
            let mut max_y = 0u32;
            for py in 0..th_u32 {
                for px in 0..tw_u32 {
                    if pixels[(py * tw_u32 + px) as usize].alpha() > 0 {
                        min_x = min_x.min(px);
                        max_x = max_x.max(px);
                        min_y = min_y.min(py);
                        max_y = max_y.max(py);
                    }
                }
            }

            if max_x >= min_x && max_y >= min_y {
                let glyph_w = (max_x - min_x + 1) as f32;
                let glyph_h = (max_y - min_y + 1) as f32;

                // Inner margin where alpha is fully opaque (same as Blur's margin)
                let inner_margin = font_size * 0.3;
                // Outer falloff zone where alpha graduates from 255 to 0.
                // Using 3*sigma covers the vast majority of a gaussian bell.
                let falloff = (*sigma * 3.0).max(font_size * 0.5);
                let total_margin = inner_margin + falloff;

                let mask_w = (glyph_w + total_margin * 2.0).ceil() as u32;
                let mask_h = (glyph_h + total_margin * 2.0).ceil() as u32;
                let mut mask_src = Pixmap::new(mask_w.max(1), mask_h.max(1))
                    .expect("Failed to allocate gradual blur mask pixmap");

                // Build a mask with gaussian-faded edges:
                // - Inside (glyph bounds + inner_margin): alpha = 255
                // - Outside that: alpha = 255 * exp(-dist² / (2 * falloff_sigma²))
                let falloff_sigma = falloff / 3.0; // so exp(-0.5 * (3σ/σ)²) ≈ 0
                let inv_2sig2 = 1.0 / (2.0 * falloff_sigma * falloff_sigma);

                // The inner rectangle within the mask where alpha stays 255
                let inner_x0 = total_margin - inner_margin;
                let inner_y0 = total_margin - inner_margin;
                let inner_x1 = inner_x0 + glyph_w + inner_margin * 2.0;
                let inner_y1 = inner_y0 + glyph_h + inner_margin * 2.0;

                for my in 0..mask_h {
                    for mx in 0..mask_w {
                        let fx = mx as f32;
                        let fy = my as f32;

                        // Distance from the inner rectangle edge (0 if inside)
                        let dx = if fx < inner_x0 {
                            inner_x0 - fx
                        } else if fx > inner_x1 {
                            fx - inner_x1
                        } else {
                            0.0
                        };
                        let dy = if fy < inner_y0 {
                            inner_y0 - fy
                        } else if fy > inner_y1 {
                            fy - inner_y1
                        } else {
                            0.0
                        };

                        let dist_sq = dx * dx + dy * dy;
                        let alpha = if dist_sq <= 0.0 {
                            255u8
                        } else {
                            (255.0 * (-dist_sq * inv_2sig2).exp()) as u8
                        };

                        if alpha > 0 {
                            let idx = (my * mask_w + mx) as usize;
                            mask_src.pixels_mut()[idx] =
                                PremultipliedColorU8::from_rgba(alpha, alpha, alpha, alpha)
                                    .unwrap();
                        }
                    }
                }

                // Position the mask so it's centered on the glyph bounds
                let mask_x = start_x + min_x as f32 - total_margin;
                let mask_y = start_y + min_y as f32 - total_margin;

                // Rotate around the same pivot as the text pixmap so blur stays aligned
                let pivot_x = start_x + text_w / 2.0 - mask_x;
                let pivot_y = start_y + text_h / 2.0 - mask_y;
                let mask_transform = Transform::from_translate(mask_x, mask_y)
                    .pre_concat(Transform::from_rotate_at(config.rotation, pivot_x, pivot_y));

                apply_blur_under_rotated_region(
                    &mut main_pixmap,
                    image,
                    &mask_src,
                    mask_transform,
                    *sigma,
                    img_w,
                    img_h,
                );
            }
        }
        TextEffect::Outline { thickness, color } => {
            // Extract alpha channel from text_pixmap
            let pixels = text_pixmap.pixels();
            let alpha_buf: Vec<u8> = pixels.iter().map(|px| px.alpha()).collect();

            // Dilate the alpha channel using a circular structuring element
            let dilated = dilate_alpha(&alpha_buf, tw_u32, th_u32, *thickness);

            // Build an outline pixmap: dilated area filled with outline color
            let mut outline_pixmap =
                Pixmap::new(tw_u32, th_u32).expect("Failed to allocate outline pixmap");
            let (or, og, ob, oa) = *color;
            for (i, out_px) in outline_pixmap.pixels_mut().iter_mut().enumerate() {
                let da = dilated[i] as u32;
                if da > 0 {
                    // Modulate outline color alpha by the dilated mask
                    let final_a = (oa as u32 * da) / 255;
                    let pr = (or as u32 * final_a) / 255;
                    let pg = (og as u32 * final_a) / 255;
                    let pb = (ob as u32 * final_a) / 255;
                    if let Some(c) =
                        PremultipliedColorU8::from_rgba(pr as u8, pg as u8, pb as u8, final_a as u8)
                    {
                        *out_px = c;
                    }
                }
            }

            // Composite outline first (behind text)
            main_pixmap.draw_pixmap(0, 0, outline_pixmap.as_ref(), &paint, transform, None);
        }
        TextEffect::Shadow {
            dx,
            dy,
            sigma,
            color,
        } => {
            // Build a shadow image from the already-rendered text alpha,
            // filled with the shadow color. This captures all glyphs
            // including fallback-rendered ones without re-rasterizing.
            let (sr, sg, sb, sa) = *color;
            let mut shadow_rgba = RgbaImage::new(tw_u32, th_u32);
            for (i, text_px) in text_pixmap.pixels().iter().enumerate() {
                let ta = text_px.alpha() as u32;
                if ta > 0 {
                    // Modulate shadow alpha by the glyph alpha
                    let final_a = ((sa as u32 * ta) / 255) as u8;
                    let x = (i as u32) % tw_u32;
                    let y = (i as u32) / tw_u32;
                    shadow_rgba.put_pixel(x, y, image::Rgba([sr, sg, sb, final_a]));
                }
            }

            // Gaussian blur the shadow to soften it
            let shadow_dyn = DynamicImage::ImageRgba8(shadow_rgba);
            let blurred_shadow = if *sigma > 0.0 {
                shadow_dyn.blur(*sigma)
            } else {
                shadow_dyn
            };
            let blurred_rgba = blurred_shadow.to_rgba8();

            // Convert blurred shadow back to a Pixmap for compositing
            let mut shadow_pixmap =
                Pixmap::new(tw_u32, th_u32).expect("Failed to allocate shadow pixmap");
            for (src, dst) in blurred_rgba.pixels().zip(shadow_pixmap.pixels_mut()) {
                *dst = ColorU8::from_rgba(src[0], src[1], src[2], src[3]).premultiply();
            }

            // Composite shadow behind text, offset by (dx, dy) in canvas space
            let shadow_transform =
                Transform::from_translate(start_x + dx, start_y + dy).pre_concat(
                    Transform::from_rotate_at(config.rotation, text_w / 2.0, text_h / 2.0),
                );
            main_pixmap.draw_pixmap(0, 0, shadow_pixmap.as_ref(), &paint, shadow_transform, None);
        }
    }

    // Composite the text on top
    main_pixmap.draw_pixmap(0, 0, text_pixmap.as_ref(), &paint, transform, None);

    let mut out_rgba = RgbaImage::new(img_w, img_h);
    for (src, dst) in main_pixmap.pixels().iter().zip(out_rgba.pixels_mut()) {
        let un_pre = src.demultiply();
        *dst = image::Rgba([un_pre.red(), un_pre.green(), un_pre.blue(), un_pre.alpha()]);
    }

    image.pixels = DynamicImage::ImageRgba8(out_rgba);
    Ok(())
}

/// Draw a single rasterized glyph onto the text pixmap with alpha blending
fn draw_glyph_pixels(
    text_pixmap: &mut Pixmap,
    tw: u32,
    th: u32,
    img: &cosmic_text::SwashImage,
    gx: f32,
    gy: f32,
    color: (u8, u8, u8, u8),
) {
    for r in 0..img.placement.height as i32 {
        for c in 0..img.placement.width as i32 {
            let px_x = gx as i32 + c;
            let px_y = gy as i32 + r;

            if px_x < 0 || px_y < 0 || px_x >= tw as i32 || px_y >= th as i32 {
                continue;
            }

            let pixel_idx = (r * img.placement.width as i32 + c) as usize;

            // Robustly handle Linux Subpixel Anti-Aliasing and Color Emojis
            let (src_r, src_g, src_b, src_a) = match img.content {
                cosmic_text::SwashContent::Mask => (color.0, color.1, color.2, img.data[pixel_idx]),
                cosmic_text::SwashContent::SubpixelMask => {
                    // Average the subpixel RGB to get a standard alpha mask
                    let a = (img.data[pixel_idx * 3] as u32
                        + img.data[pixel_idx * 3 + 1] as u32
                        + img.data[pixel_idx * 3 + 2] as u32)
                        / 3;
                    (color.0, color.1, color.2, a as u8)
                }
                cosmic_text::SwashContent::Color => {
                    let start = pixel_idx * 4;
                    (
                        img.data[start],
                        img.data[start + 1],
                        img.data[start + 2],
                        img.data[start + 3],
                    )
                }
            };

            let target_alpha = color.3 as u32;
            let final_alpha = (src_a as u32 * target_alpha) / 255;

            if final_alpha > 0 {
                let pr = (src_r as u32 * final_alpha) / 255;
                let pg = (src_g as u32 * final_alpha) / 255;
                let pb = (src_b as u32 * final_alpha) / 255;

                let idx = (px_y as u32 * tw + px_x as u32) as usize;
                let dst = text_pixmap.pixels_mut()[idx];

                let inv_alpha = 255 - final_alpha;
                let out_a = final_alpha + (dst.alpha() as u32 * inv_alpha) / 255;
                let out_r = pr + (dst.red() as u32 * inv_alpha) / 255;
                let out_g = pg + (dst.green() as u32 * inv_alpha) / 255;
                let out_b = pb + (dst.blue() as u32 * inv_alpha) / 255;

                if let Some(blended) = PremultipliedColorU8::from_rgba(
                    out_r as u8,
                    out_g as u8,
                    out_b as u8,
                    out_a as u8,
                ) {
                    text_pixmap.pixels_mut()[idx] = blended;
                }
            }
        }
    }
}
