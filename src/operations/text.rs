use crate::{arg_parse_err::ArgParseErr, error::MagickError, image::Image, wm_err};
use cosmic_text::{Align, Attrs, Buffer, Family, FontSystem, Metrics, Shaping, SwashCache};
use image::{DynamicImage, GrayImage, ImageBuffer, RgbaImage};
use qrcode::{EcLevel, QrCode};
use rayon::prelude::*;
use std::sync::{Mutex, OnceLock};
use tiny_skia::{
    BlendMode, ColorU8, FilterQuality, Pixmap, PixmapPaint, PremultipliedColorU8, Transform,
};

/// Upper bound on any intermediate layer dimension. Guards against a huge or
/// non-finite computed size turning into a bogus allocation via the saturating
/// `f32 as u32` cast.
const MAX_LAYER_DIM: u32 = 100_000;

/// A parsed QR code block extracted from the text field.
#[derive(Debug, Clone, PartialEq)]
struct QrBlock {
    pub ec_level: EcLevel,
    pub content: String,
    /// Gaussian blur sigma for the region under the QR code (0.0 = disabled).
    pub blur_sigma: f32,
    /// Gradual blur extent as a percentage of QR width/height (0.0 = sharp edges).
    /// E.g. 15.0 means the blur fades out over an additional 15% beyond the QR block.
    pub blur_gradual_pct: f32,
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
/// and escape sequences (`\{`, `\\`, and `\}` within QR content).
///
/// - `{QR:L:0:#00000000:hello}` → QR block, no blur, transparent light color
/// - `{QR:H:3.0:#FFFFFFCC:https://x.com}` → QR block, blur sigma=3.0 (sharp edges)
/// - `{QR:H:100+15:#FFFFFFCC:https://x.com}` → QR block, blur sigma=100, gradual 15% of QR size
/// - `\{QR:L:...}` → literal text "{QR:L:...}"
fn parse_text_segments(text: &str) -> Result<Vec<TextSegment>, ArgParseErr> {
    let mut segments: Vec<TextSegment> = Vec::new();
    let mut plain = String::new();
    // Walk byte offsets directly instead of collecting into a `Vec<char>`.
    // `str::find` below returns byte offsets; mixing those with char indices
    // used to advance the cursor too far and silently swallow the characters
    // following any QR block whose content contained non-ASCII text. Byte
    // indexing also avoids re-collecting the tail of the string on every brace.
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        // Handle escapes: \\ -> literal \, \{ -> literal {
        if bytes[i] == b'\\' {
            if i + 1 < len {
                if bytes[i + 1] == b'\\' {
                    plain.push('\\');
                    i += 2;
                    continue;
                }
                if bytes[i + 1] == b'{' {
                    plain.push('{');
                    i += 2;
                    continue;
                }
            }
            // Not a supported escape, keep backslash as-is
            plain.push('\\');
            i += 1;
            continue;
        }

        // Detect {QR:...:...:...:...}
        if bytes[i] == b'{' {
            let rest = &text[i..];
            if rest.starts_with("{QR:") {
                // Find the closing brace, respecting escapes inside the QR block
                let mut close_pos = None;
                let mut escaped = false;
                for (offset, ch) in rest[4..].char_indices() {
                    if escaped {
                        escaped = false;
                    } else if ch == '\\' {
                        escaped = true;
                    } else if ch == '}' {
                        close_pos = Some(4 + offset);
                        break;
                    }
                }

                let close_pos = close_pos
                    .ok_or_else(|| ArgParseErr::with_msg("unclosed QR block: missing '}'"))?;

                let inner = &rest[4..close_pos]; // after "{QR:" and before "}"

                // Parse: EcLevel:blur_sigma:light_color:content
                // Split at first 3 colons to get 4 parts.
                // The first `next()` on a `SplitN` always yields a value
                // (possibly empty), so only the later fields can be absent.
                let mut parts = inner.splitn(4, ':');
                let ec_str = parts.next().unwrap_or("");
                let blur_str = parts
                    .next()
                    .ok_or_else(|| ArgParseErr::with_msg("QR block missing blur sigma"))?;
                let light_str = parts
                    .next()
                    .ok_or_else(|| ArgParseErr::with_msg("QR block missing light_color"))?;
                let content_raw = parts
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

                // Parse blur field: either "sigma" (sharp) or "sigma+pct" (gradual).
                // E.g. "100" → sigma=100, sharp edges.
                //      "100+15" → sigma=100, gradual 15% of QR size.
                let (blur_sigma, blur_gradual_pct) = if let Some(plus_pos) = blur_str.find('+') {
                    let sigma_str = &blur_str[..plus_pos];
                    let pct_str = &blur_str[plus_pos + 1..];
                    let sigma = sigma_str.parse::<f32>().map_err(|_| {
                        ArgParseErr::with_msg("QR blur sigma must be a number (e.g. '100+15')")
                    })?;
                    let pct = pct_str.parse::<f32>().map_err(|_| {
                        ArgParseErr::with_msg(
                            "QR blur gradual percentage must be a number (e.g. '100+15')",
                        )
                    })?;
                    (sigma, pct)
                } else {
                    let sigma = blur_str.parse::<f32>().map_err(|_| {
                        ArgParseErr::with_msg(
                            "QR blur must be a number (0 to disable, e.g. 3.0) \
                             or sigma+pct for gradual blur (e.g. 100+15)",
                        )
                    })?;
                    (sigma, 0.0)
                };

                // Reject values the renderer can't act on rather than
                // letting NaN/negatives reach the blur kernel.
                if !blur_sigma.is_finite() || blur_sigma < 0.0 {
                    return Err(ArgParseErr::with_msg(
                        "QR blur sigma must be a finite number >= 0",
                    ));
                }
                if !blur_gradual_pct.is_finite() || blur_gradual_pct < 0.0 {
                    return Err(ArgParseErr::with_msg(
                        "QR blur gradual percentage must be a finite number >= 0",
                    ));
                }

                let light_color = parse_hex_color(light_str)?;

                // Unescape content (e.g. \} -> }, \\ -> \)
                let mut content = String::with_capacity(content_raw.len());
                let mut chars = content_raw.chars().peekable();
                while let Some(c) = chars.next() {
                    if c == '\\' {
                        if let Some(&next) = chars.peek()
                            && (next == '}' || next == '\\') {
                                content.push(next);
                                chars.next();
                                continue;
                            }
                        content.push('\\');
                    } else {
                        content.push(c);
                    }
                }

                // Flush accumulated plain text
                if !plain.is_empty() {
                    segments.push(TextSegment::Plain(std::mem::take(&mut plain)));
                }

                segments.push(TextSegment::Qr(QrBlock {
                    ec_level,
                    content,
                    blur_sigma,
                    blur_gradual_pct,
                    light_color,
                }));

                i += close_pos + 1; // skip past '}' (both are byte offsets)
                continue;
            }
        }

        // Copy one whole character, not one byte, so multi-byte text survives.
        match text[i..].chars().next() {
            Some(ch) => {
                plain.push(ch);
                i += ch.len_utf8();
            }
            None => break,
        }
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
            let val = pct
                .parse::<f32>()
                .map_err(|_| ArgParseErr::with_msg("invalid percentage position"))?;
            if !val.is_finite() || !(0.0..=100.0).contains(&val) {
                return Err(ArgParseErr::with_msg(
                    "position percentage must be between 0% and 100%",
                ));
            }
            Ok(Position::Percent(val))
        } else if let Some(em) = s.strip_suffix("em") {
            Ok(Position::Em(finite(
                em.parse::<f32>().map_err(|_| ArgParseErr::with_msg("invalid em position"))?,
                "position must be a finite number",
            )?))
        } else {
            Ok(Position::Absolute(finite(
                s.parse::<f32>().map_err(|_| ArgParseErr::with_msg("invalid absolute position"))?,
                "position must be a finite number",
            )?))
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

/// Reject NaN and infinities at parse time so they can't propagate into layer
/// sizes, transforms or blur kernels.
fn finite(v: f32, msg: &'static str) -> Result<f32, ArgParseErr> {
    if v.is_finite() { Ok(v) } else { Err(ArgParseErr::with_msg(msg)) }
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
    Outline { thickness: u32, color: (u8, u8, u8, u8) },
    /// Soft drop shadow behind the text glyphs, offset by (dx, dy) pixels
    /// and gaussian-blurred to give depth.
    Shadow { dx: f32, dy: f32, sigma: f32, color: (u8, u8, u8, u8) },
    /// Combination of a background blur effect and a glyph effect (outline or shadow).
    Combined { bg: Box<TextEffect>, glyph: Box<TextEffect> },
}

/// Optional pixel-displacement layer that runs *after* the base `TextEffect`,
/// warping the (already-effected) background before the text is composited.
/// Append with `+` in the effect string, e.g. `blur:5.0+explode:30`.
#[derive(Debug, Clone, PartialEq)]
pub enum DisplacementEffect {
    None,
    /// Radial push from the text centroid.
    /// `binding` controls outline/shadow follow-up strength (0.0–1.0, default 1.0).
    Explode {
        strength: f32,
        binding: f32,
    },
    /// Text-shape-driven displacement via a proximity field.
    /// `direction` = -1 → omnidirectional; 0–360 → fixed bearing in degrees
    /// (0 = right, 90 = down, 180 = left, 270 = up).
    /// `binding` controls outline/shadow follow-up strength (0.0–1.0, default 1.0).
    Meltdown {
        strength: f32,
        direction: f32,
        binding: f32,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct TextConfig {
    pub effect: TextEffect,
    pub displacement: DisplacementEffect,
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
    let hex = hex.trim();
    // `strip_prefix` removes exactly one '#'; `trim_start_matches` used to eat
    // any number of them, silently accepting "###FFFFFF".
    let hex = hex.strip_prefix('#').unwrap_or(hex);
    // The slices below are byte slices, so a non-ASCII string of the right byte
    // length would panic on a char boundary (e.g. "#日本" is 6 bytes).
    if !hex.is_ascii() {
        return Err(ArgParseErr::with_msg("color must be #RRGGBB or #RRGGBBAA"));
    }
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
    /// Effect values (base):
    ///   - `none`                 — plain text, no background effect
    ///   - `blur:5.0`             — blur background behind text (sigma=5.0)
    ///   - `gradualblur:5.0`      — blur with smooth graduated edges (sigma=5.0)
    ///   - `outline:3:#000000FF`  — subtitle-style outline (thickness 3, black)
    ///   - `shadow:3:3:4.0:#00000080` — drop shadow (dx, dy, sigma, color)
    ///
    /// Combined effects (join background blur + glyph effect with `+`):
    ///   - `blur:5.0+outline:3:#000000FF` — blur background and outline text
    ///   - `gradualblur:5.0+shadow:3:3:4.0:#00000080` — gradual blur + drop shadow
    ///
    /// Displacement modifiers (append with `+`):
    ///   - `+explode:30`          — radial pixel explosion (strength in px, binding default 1.0)
    ///   - `+explode:30:0.5`      — radial pixel explosion with binding=0.5
    ///   - `+meltdown:40:-1`      — text-shaped blast, omnidirectional (binding default 1.0)
    ///   - `+meltdown:40:90:0.8`  — text-shaped blast downward with binding=0.8
    ///   - `+binding:0.5`         — explicit modifier flag setting outline/shadow follow-up strength (0.0–1.0)
    ///
    /// Combined examples:
    ///   - `blur:5.0+explode:30`  — blur then explode
    ///   - `outline:3:#000000FF+meltdown:40:-1` — outline with meltdown
    ///   - `blur:5.0+outline:3:#000000FF+explode:30+binding:0.5` — blur + outline + explode with 50% binding
    ///   - `explode:30`           — plain text with explode (implicit `none` base)
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

        let (effect, displacement) = Self::parse_effect_field(effect_str)?;

        let mut parts: Vec<&str> = rest.rsplitn(8, ',').collect();
        if parts.len() != 8 {
            return Err(ArgParseErr::with_msg(
                "text requires 9 comma-separated values: \
                 effect,text,font_name,font_size,color,rotation,justify,x,y",
            ));
        }
        parts.reverse();

        let text = parts[0].replace("\\n", "\n");
        // Validate any embedded QR blocks now rather than failing halfway
        // through rendering. The parsed segments are rebuilt at render time.
        parse_text_segments(&text)?;

        let font_name = parts[1].trim().to_string();

        let size_str = parts[2].trim();
        let font_size = if let Some(pct) = size_str.strip_suffix('%') {
            let v = pct.parse::<f32>().map_err(|_| ArgParseErr::with_msg("invalid pct size"))?;
            if !v.is_finite() || v <= 0.0 || v > 1000.0 {
                return Err(ArgParseErr::with_msg("font size percentage must be > 0 and <= 1000%"));
            }
            FontSize::RelativePercent(v)
        } else {
            let v =
                size_str.parse::<f32>().map_err(|_| ArgParseErr::with_msg("invalid abs size"))?;
            if !v.is_finite() || v <= 0.0 || v > 20_000.0 {
                return Err(ArgParseErr::with_msg("absolute font size must be > 0 and <= 20000"));
            }
            FontSize::Absolute(v)
        };

        let color = parse_hex_color(parts[3])?;

        let rotation = finite(
            parts[4]
                .trim()
                .parse::<f32>()
                .map_err(|_| ArgParseErr::with_msg("invalid rotation (must be float degrees)"))?,
            "rotation must be a finite number of degrees",
        )?;

        let justify = match parts[5].trim().to_lowercase().as_str() {
            "left" => Align::Left,
            "center" | "middle" => Align::Center,
            "right" => Align::Right,
            _ => {
                return Err(ArgParseErr::with_msg(
                    "justify must be left, center (or middle), or right",
                ));
            }
        };

        let x = Position::parse(parts[6])?;
        let y = Position::parse(parts[7])?;

        Ok(Self {
            effect,
            displacement,
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

    /// Split on `+`, parsing background effects, glyph effects, and optional displacement modifiers.
    /// Supports combining background blur (blur/gradualblur) with glyph effects (outline/shadow).
    /// Standalone `explode:…` / `meltdown:…` (no `+`) is accepted as an
    /// implicit `none` base for backward compatibility.
    fn parse_effect_field(s: &str) -> Result<(TextEffect, DisplacementEffect), ArgParseErr> {
        let s = s.trim();
        if s.is_empty() || s.eq_ignore_ascii_case("none") {
            return Ok((TextEffect::None, DisplacementEffect::None));
        }

        const SEPARATOR_PREFIXES: &[&str] = &[
            "blur:",
            "gradualblur:",
            "outline:",
            "shadow:",
            "explode:",
            "meltdown:",
            "binding:",
            "none",
        ];

        let mut parts = Vec::new();
        let mut last_start = 0;
        let bytes = s.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'+' {
                let after = &s[i + 1..];
                let is_sep = SEPARATOR_PREFIXES.iter().any(|prefix| {
                    after.len() >= prefix.len()
                        && after[..prefix.len()].eq_ignore_ascii_case(prefix)
                });
                if is_sep {
                    parts.push(&s[last_start..i]);
                    last_start = i + 1;
                }
            }
            i += 1;
        }
        parts.push(&s[last_start..]);

        let mut bg_effect: Option<TextEffect> = None;
        let mut glyph_effect: Option<TextEffect> = None;
        let mut displacement: Option<DisplacementEffect> = None;
        let mut explicit_binding: Option<f32> = None;

        for part in parts {
            let part = part.trim();
            if part.is_empty() || part.eq_ignore_ascii_case("none") {
                continue;
            }

            if let Some(binding_str) = part.strip_prefix("binding:") {
                let b = binding_str.parse::<f32>().map_err(|_| {
                    ArgParseErr::with_msg(
                        "binding must be a float between 0.0 and 1.0 (e.g. 'binding:0.5')",
                    )
                })?;
                if !b.is_finite() || !(0.0..=1.0).contains(&b) {
                    return Err(ArgParseErr::with_msg("binding must be between 0.0 and 1.0"));
                }
                explicit_binding = Some(b);
            } else if part.starts_with("explode:") || part.starts_with("meltdown:") {
                if displacement.is_some() {
                    return Err(ArgParseErr::with_msg("multiple displacement modifiers specified"));
                }
                displacement = Some(Self::parse_displacement(part)?);
            } else if part.starts_with("blur:") || part.starts_with("gradualblur:") {
                if bg_effect.is_some() {
                    return Err(ArgParseErr::with_msg(
                        "multiple background blur effects specified",
                    ));
                }
                bg_effect = Some(Self::parse_base_effect(part)?);
            } else if part.starts_with("outline:") || part.starts_with("shadow:") {
                if glyph_effect.is_some() {
                    return Err(ArgParseErr::with_msg(
                        "multiple glyph effects specified (outline/shadow)",
                    ));
                }
                glyph_effect = Some(Self::parse_base_effect(part)?);
            } else {
                return Err(ArgParseErr::with_msg(
                    "effect must be 'none', 'blur:<sigma>', 'gradualblur:<sigma>', \
                     'outline:<thickness>:<#color>', 'shadow:<dx>:<dy>:<sigma>:<#color>', \
                     'explode:<strength>[:<binding>]', 'meltdown:<strength>:<direction>[:<binding>]', \
                     or 'binding:<0.0-1.0>'",
                ));
            }
        }

        let effect = match (bg_effect, glyph_effect) {
            (Some(bg), Some(glyph)) => {
                TextEffect::Combined { bg: Box::new(bg), glyph: Box::new(glyph) }
            }
            (Some(bg), None) => bg,
            (None, Some(glyph)) => glyph,
            (None, None) => TextEffect::None,
        };

        let mut disp = displacement.unwrap_or(DisplacementEffect::None);
        if let Some(b) = explicit_binding {
            match &mut disp {
                DisplacementEffect::Explode { binding, .. } => *binding = b,
                DisplacementEffect::Meltdown { binding, .. } => *binding = b,
                DisplacementEffect::None => {
                    return Err(ArgParseErr::with_msg(
                        "binding modifier specified without explode or meltdown displacement",
                    ));
                }
            }
        }

        Ok((effect, disp))
    }

    fn parse_base_effect(s: &str) -> Result<TextEffect, ArgParseErr> {
        let s = s.trim();
        if s.eq_ignore_ascii_case("none") {
            return Ok(TextEffect::None);
        }
        if let Some(sigma_str) = s.strip_prefix("blur:") {
            let sigma = sigma_str.parse::<f32>().map_err(|_| {
                ArgParseErr::with_msg("blur effect requires a float sigma (e.g. 'blur:5.0')")
            })?;
            // A zero or negative sigma produces a degenerate gaussian kernel,
            // so require a real one instead of silently blanking the region.
            if !sigma.is_finite() || sigma <= 0.0 {
                return Err(ArgParseErr::with_msg(
                    "blur sigma must be a finite number > 0 (use 'none' to disable)",
                ));
            }
            return Ok(TextEffect::Blur { sigma });
        }
        if let Some(sigma_str) = s.strip_prefix("gradualblur:") {
            let sigma = sigma_str.parse::<f32>().map_err(|_| {
                ArgParseErr::with_msg(
                    "gradualblur effect requires a float sigma (e.g. 'gradualblur:5.0')",
                )
            })?;
            if !sigma.is_finite() || sigma <= 0.0 {
                return Err(ArgParseErr::with_msg(
                    "gradualblur sigma must be a finite number > 0 (use 'none' to disable)",
                ));
            }
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
            if thickness > MAX_LAYER_DIM {
                return Err(ArgParseErr::with_msg("outline thickness is unreasonably large"));
            }
            let color = parse_hex_color(color_str)?;
            return Ok(TextEffect::Outline { thickness, color });
        }
        if let Some(rest) = s.strip_prefix("shadow:") {
            // Format: shadow:dx:dy:sigma:#color
            // The first `next()` always yields a value, so only the later
            // fields can actually be missing.
            let mut parts = rest.splitn(4, ':');
            let dx_str = parts.next().unwrap_or("");
            let dy_str =
                parts.next().ok_or_else(|| ArgParseErr::with_msg("shadow effect missing dy"))?;
            let sigma_str =
                parts.next().ok_or_else(|| ArgParseErr::with_msg("shadow effect missing sigma"))?;
            let color_str =
                parts.next().ok_or_else(|| ArgParseErr::with_msg("shadow effect missing color"))?;
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
            // sigma == 0 is legal here and means a hard-edged shadow; the
            // renderer skips the blur in that case.
            if !dx.is_finite() || !dy.is_finite() {
                return Err(ArgParseErr::with_msg("shadow dx/dy must be finite numbers"));
            }
            if !sigma.is_finite() || sigma < 0.0 {
                return Err(ArgParseErr::with_msg("shadow sigma must be a finite number >= 0"));
            }
            let color = parse_hex_color(color_str)?;
            return Ok(TextEffect::Shadow { dx, dy, sigma, color });
        }
        Err(ArgParseErr::with_msg(
            "base effect must be 'none', 'blur:<sigma>', 'gradualblur:<sigma>', \
             'outline:<thickness>:<#color>', or 'shadow:<dx>:<dy>:<sigma>:<#color>'",
        ))
    }

    fn parse_displacement(s: &str) -> Result<DisplacementEffect, ArgParseErr> {
        let s = s.trim();
        if let Some(rest) = s.strip_prefix("explode:") {
            let mut parts = rest.split(':');
            let strength_str = parts.next().unwrap_or("");
            let strength = strength_str.parse::<f32>().map_err(|_| {
                ArgParseErr::with_msg("explode requires a float strength (e.g. 'explode:30')")
            })?;
            if !strength.is_finite() || strength <= 0.0 {
                return Err(ArgParseErr::with_msg("explode strength must be a finite number > 0"));
            }
            let binding = if let Some(b_str) = parts.next() {
                let b = b_str.parse::<f32>().map_err(|_| {
                    ArgParseErr::with_msg("explode binding must be a float between 0.0 and 1.0")
                })?;
                if !b.is_finite() || !(0.0..=1.0).contains(&b) {
                    return Err(ArgParseErr::with_msg(
                        "explode binding must be between 0.0 and 1.0",
                    ));
                }
                b
            } else {
                1.0
            };
            return Ok(DisplacementEffect::Explode { strength, binding });
        }
        if let Some(rest) = s.strip_prefix("meltdown:") {
            // Format: meltdown:strength:direction[:binding]
            let mut parts = rest.split(':');
            let strength_str = parts.next().unwrap_or("");
            let dir_str = parts.next().ok_or_else(|| {
                ArgParseErr::with_msg(
                    "meltdown requires strength:direction \
                     (e.g. 'meltdown:40:-1' or 'meltdown:40:90')",
                )
            })?;
            let strength = strength_str
                .parse::<f32>()
                .map_err(|_| ArgParseErr::with_msg("meltdown strength must be a float"))?;
            let direction = dir_str.parse::<f32>().map_err(|_| {
                ArgParseErr::with_msg("meltdown direction must be -1 (omni) or 0-360 (degrees)")
            })?;
            if !strength.is_finite() || strength <= 0.0 {
                return Err(ArgParseErr::with_msg("meltdown strength must be a finite number > 0"));
            }
            if !direction.is_finite() || (direction != -1.0 && !(0.0..=360.0).contains(&direction))
            {
                return Err(ArgParseErr::with_msg(
                    "meltdown direction must be -1 (omni) or 0-360 (degrees)",
                ));
            }
            let binding = if let Some(b_str) = parts.next() {
                let b = b_str.parse::<f32>().map_err(|_| {
                    ArgParseErr::with_msg("meltdown binding must be a float between 0.0 and 1.0")
                })?;
                if !b.is_finite() || !(0.0..=1.0).contains(&b) {
                    return Err(ArgParseErr::with_msg(
                        "meltdown binding must be between 0.0 and 1.0",
                    ));
                }
                b
            } else {
                1.0
            };
            return Ok(DisplacementEffect::Meltdown { strength, direction, binding });
        }
        Err(ArgParseErr::with_msg(
            "displacement modifier must be 'explode:<strength>[:<binding>]' \
             or 'meltdown:<strength>:<direction>[:<binding>]'",
        ))
    }
}

/// Shared, lazily-built font state.
///
/// `FontSystem::new()` enumerates and parses every font installed on the
/// system, which costs hundreds of milliseconds. Building one per call made
/// that the dominant cost of rendering a single line of text, and it also threw
/// away the glyph raster cache every time. Both now live behind a process-wide
/// `OnceLock`; the `Mutex` is needed because cosmic-text's APIs take `&mut`.
struct SharedFonts {
    font_system: FontSystem,
    swash_cache: SwashCache,
}

static SHARED_FONTS: OnceLock<Mutex<SharedFonts>> = OnceLock::new();

fn shared_fonts() -> &'static Mutex<SharedFonts> {
    SHARED_FONTS.get_or_init(|| {
        Mutex::new(SharedFonts { font_system: FontSystem::new(), swash_cache: SwashCache::new() })
    })
}

/// Allocate a pixmap, turning tiny_skia's `None` (zero-sized, or too large to
/// address) into a proper error instead of a panic in the middle of a render.
fn new_pixmap(w: u32, h: u32, what: &str) -> Result<Pixmap, MagickError> {
    Pixmap::new(w, h).ok_or_else(|| wm_err!("failed to allocate {} pixmap ({}x{})", what, w, h))
}

/// Convert a computed layer size in pixels to a dimension, rejecting NaN and
/// overflow rather than letting the saturating `f32 as u32` cast turn them into
/// a `u32::MAX`-sized allocation request.
fn layer_dim(v: f32, what: &str) -> Result<u32, MagickError> {
    if !v.is_finite() || v < 0.0 || v > MAX_LAYER_DIM as f32 {
        return Err(wm_err!("computed {} layer size is out of range: {}", what, v));
    }
    Ok((v.ceil() as u32).max(1))
}

/// Scan an alpha channel to find the tight axis-aligned bounding box of all
/// non-transparent pixels. Returns `(min_x, max_x, min_y, max_y)` or `None` if
/// every pixel is fully transparent. Used both for the rendered glyphs and for
/// the projected blur mask footprint.
fn compute_alpha_bbox(
    pixels: &[PremultipliedColorU8],
    w: u32,
    h: u32,
) -> Option<(u32, u32, u32, u32)> {
    let mut min_x = w;
    let mut max_x = 0u32;
    let mut min_y = h;
    let mut max_y = 0u32;
    let mut any = false;
    for py in 0..h {
        for px in 0..w {
            if pixels[(py * w + px) as usize].alpha() > 0 {
                any = true;
                min_x = min_x.min(px);
                max_x = max_x.max(px);
                min_y = min_y.min(py);
                max_y = max_y.max(py);
            }
        }
    }
    if any { Some((min_x, max_x, min_y, max_y)) } else { None }
}

fn same_pixel(a: PremultipliedColorU8, b: PremultipliedColorU8) -> bool {
    a.red() == b.red() && a.green() == b.green() && a.blue() == b.blue() && a.alpha() == b.alpha()
}

/// What we need to remember about the source image in order to write the
/// rendered result back without silently degrading it.
///
/// tiny_skia composites in 8-bit premultiplied RGBA, so the drawing itself is
/// necessarily 8-bit. Keep the original source image and only replace the
/// pixels the pipeline actually modified, preserving the original precision and
/// layout for untouched areas.
struct SourcePrecision {
    orig: DynamicImage,
    /// The 8-bit starting point, used to detect which pixels changed.
    base: Vec<PremultipliedColorU8>,
}

/// Build the 8-bit working pixmap for the tiny_skia pipeline and capture what's
/// needed to restore the source's precision afterwards.
fn begin_render(image: &Image) -> Result<(Pixmap, SourcePrecision), MagickError> {
    let (w, h) = (image.pixels.width(), image.pixels.height());
    let mut pixmap = new_pixmap(w, h, "main")?;

    match &image.pixels {
        DynamicImage::ImageLuma8(img) => {
            for (src, dst) in img.pixels().iter().zip(pixmap.pixels_mut().iter_mut()) {
                *dst = ColorU8::from_rgba(src[0], src[0], src[0], 255).premultiply();
            }
        }
        DynamicImage::ImageLumaA8(img) => {
            for (src, dst) in img.pixels().iter().zip(pixmap.pixels_mut().iter_mut()) {
                *dst = ColorU8::from_rgba(src[0], src[0], src[0], src[1]).premultiply();
            }
        }
        DynamicImage::ImageRgb8(img) => {
            for (src, dst) in img.pixels().iter().zip(pixmap.pixels_mut().iter_mut()) {
                *dst = ColorU8::from_rgba(src[0], src[1], src[2], 255).premultiply();
            }
        }
        DynamicImage::ImageRgba8(img) => {
            for (src, dst) in img.pixels().iter().zip(pixmap.pixels_mut().iter_mut()) {
                *dst = ColorU8::from_rgba(src[0], src[1], src[2], src[3]).premultiply();
            }
        }
        DynamicImage::ImageLuma16(img) => {
            for (src, dst) in img.pixels().iter().zip(pixmap.pixels_mut().iter_mut()) {
                let l = ((src[0] as u32 + 128) / 257) as u8;
                *dst = ColorU8::from_rgba(l, l, l, 255).premultiply();
            }
        }
        DynamicImage::ImageLumaA16(img) => {
            for (src, dst) in img.pixels().iter().zip(pixmap.pixels_mut().iter_mut()) {
                let l = ((src[0] as u32 + 128) / 257) as u8;
                let a = ((src[1] as u32 + 128) / 257) as u8;
                *dst = ColorU8::from_rgba(l, l, l, a).premultiply();
            }
        }
        DynamicImage::ImageRgb16(img) => {
            for (src, dst) in img.pixels().iter().zip(pixmap.pixels_mut().iter_mut()) {
                let r = ((src[0] as u32 + 128) / 257) as u8;
                let g = ((src[1] as u32 + 128) / 257) as u8;
                let b = ((src[2] as u32 + 128) / 257) as u8;
                *dst = ColorU8::from_rgba(r, g, b, 255).premultiply();
            }
        }
        DynamicImage::ImageRgba16(img) => {
            for (src, dst) in img.pixels().iter().zip(pixmap.pixels_mut().iter_mut()) {
                let r = ((src[0] as u32 + 128) / 257) as u8;
                let g = ((src[1] as u32 + 128) / 257) as u8;
                let b = ((src[2] as u32 + 128) / 257) as u8;
                let a = ((src[3] as u32 + 128) / 257) as u8;
                *dst = ColorU8::from_rgba(r, g, b, a).premultiply();
            }
        }
        DynamicImage::ImageRgb32F(img) => {
            for (src, dst) in img.pixels().iter().zip(pixmap.pixels_mut().iter_mut()) {
                let r = (src[0].clamp(0.0, 1.0) * 255.0).round() as u8;
                let g = (src[1].clamp(0.0, 1.0) * 255.0).round() as u8;
                let b = (src[2].clamp(0.0, 1.0) * 255.0).round() as u8;
                *dst = ColorU8::from_rgba(r, g, b, 255).premultiply();
            }
        }
        DynamicImage::ImageRgba32F(img) => {
            for (src, dst) in img.pixels().iter().zip(pixmap.pixels_mut().iter_mut()) {
                let r = (src[0].clamp(0.0, 1.0) * 255.0).round() as u8;
                let g = (src[1].clamp(0.0, 1.0) * 255.0).round() as u8;
                let b = (src[2].clamp(0.0, 1.0) * 255.0).round() as u8;
                let a = (src[3].clamp(0.0, 1.0) * 255.0).round() as u8;
                *dst = ColorU8::from_rgba(r, g, b, a).premultiply();
            }
        }
        _ => {
            let rgba = image.pixels.to_rgba8();
            for (src, dst) in rgba.pixels().iter().zip(pixmap.pixels_mut().iter_mut()) {
                *dst = ColorU8::from_rgba(src[0], src[1], src[2], src[3]).premultiply();
            }
        }
    }

    let base = pixmap.pixels().to_vec();
    Ok((pixmap, SourcePrecision { orig: image.pixels.clone(), base }))
}

/// Store the finished pixmap back into the image at the source's bit depth.
fn finish_render(image: &mut Image, pixmap: &Pixmap, precision: SourcePrecision) {
    let (w, h) = (pixmap.width(), pixmap.height());
    match precision.orig {
        DynamicImage::ImageLuma8(orig) => {
            let mut can_stay_luma = true;
            let mut can_stay_rgb = true;

            for (i, res) in pixmap.pixels().iter().enumerate() {
                if !same_pixel(*res, precision.base[i]) {
                    let un_pre = res.demultiply();
                    if un_pre.alpha() != 255 {
                        can_stay_luma = false;
                        can_stay_rgb = false;
                        break;
                    }
                    if un_pre.red() != un_pre.green() || un_pre.green() != un_pre.blue() {
                        can_stay_luma = false;
                    }
                }
            }

            if can_stay_luma {
                let mut out = GrayImage::new(w, h);
                for (i, ((res, orig_px), dst)) in pixmap
                    .pixels()
                    .iter()
                    .zip(orig.pixels().iter())
                    .zip(out.pixels_mut().iter_mut())
                    .enumerate()
                {
                    if same_pixel(*res, precision.base[i]) {
                        *dst = *orig_px;
                    } else {
                        let un_pre = res.demultiply();
                        *dst = image::Luma([un_pre.red()]);
                    }
                }
                image.pixels = DynamicImage::ImageLuma8(out);
            } else if can_stay_rgb {
                let mut rgb = image::RgbImage::new(w, h);
                for (i, ((res, orig_px), dst)) in pixmap
                    .pixels()
                    .iter()
                    .zip(orig.pixels().iter())
                    .zip(rgb.pixels_mut().iter_mut())
                    .enumerate()
                {
                    if same_pixel(*res, precision.base[i]) {
                        *dst = image::Rgb([orig_px[0], orig_px[0], orig_px[0]]);
                    } else {
                        let un_pre = res.demultiply();
                        *dst = image::Rgb([un_pre.red(), un_pre.green(), un_pre.blue()]);
                    }
                }
                image.pixels = DynamicImage::ImageRgb8(rgb);
            } else {
                let mut rgba = RgbaImage::new(w, h);
                for (i, ((res, orig_px), dst)) in pixmap
                    .pixels()
                    .iter()
                    .zip(orig.pixels().iter())
                    .zip(rgba.pixels_mut().iter_mut())
                    .enumerate()
                {
                    if same_pixel(*res, precision.base[i]) {
                        *dst = image::Rgba([orig_px[0], orig_px[0], orig_px[0], 255]);
                    } else {
                        let un_pre = res.demultiply();
                        *dst = image::Rgba([
                            un_pre.red(),
                            un_pre.green(),
                            un_pre.blue(),
                            un_pre.alpha(),
                        ]);
                    }
                }
                image.pixels = DynamicImage::ImageRgba8(rgba);
            }
        }
        DynamicImage::ImageLumaA8(orig) => {
            let mut can_stay_luma = true;
            for (i, res) in pixmap.pixels().iter().enumerate() {
                if !same_pixel(*res, precision.base[i]) {
                    let un_pre = res.demultiply();
                    if un_pre.red() != un_pre.green() || un_pre.green() != un_pre.blue() {
                        can_stay_luma = false;
                        break;
                    }
                }
            }
            if can_stay_luma {
                let mut out = ImageBuffer::<image::LumaA<u8>, Vec<u8>>::new(w, h);
                for (i, ((res, orig_px), dst)) in pixmap
                    .pixels()
                    .iter()
                    .zip(orig.pixels().iter())
                    .zip(out.pixels_mut().iter_mut())
                    .enumerate()
                {
                    if same_pixel(*res, precision.base[i]) {
                        *dst = *orig_px;
                    } else {
                        let un_pre = res.demultiply();
                        *dst = image::LumaA([un_pre.red(), un_pre.alpha()]);
                    }
                }
                image.pixels = DynamicImage::ImageLumaA8(out);
            } else {
                let mut rgba = RgbaImage::new(w, h);
                for (i, ((res, orig_px), dst)) in pixmap
                    .pixels()
                    .iter()
                    .zip(orig.pixels().iter())
                    .zip(rgba.pixels_mut().iter_mut())
                    .enumerate()
                {
                    if same_pixel(*res, precision.base[i]) {
                        *dst = image::Rgba([orig_px[0], orig_px[0], orig_px[0], orig_px[1]]);
                    } else {
                        let un_pre = res.demultiply();
                        *dst = image::Rgba([
                            un_pre.red(),
                            un_pre.green(),
                            un_pre.blue(),
                            un_pre.alpha(),
                        ]);
                    }
                }
                image.pixels = DynamicImage::ImageRgba8(rgba);
            }
        }
        DynamicImage::ImageRgb8(orig) => {
            let mut opaque = true;
            for (i, res) in pixmap.pixels().iter().enumerate() {
                if !same_pixel(*res, precision.base[i])
                    && res.alpha() != 255 {
                        opaque = false;
                        break;
                    }
            }
            if opaque {
                let mut rgb = image::RgbImage::new(w, h);
                for (i, ((res, orig_px), dst)) in pixmap
                    .pixels()
                    .iter()
                    .zip(orig.pixels().iter())
                    .zip(rgb.pixels_mut().iter_mut())
                    .enumerate()
                {
                    if same_pixel(*res, precision.base[i]) {
                        *dst = *orig_px;
                    } else {
                        let un_pre = res.demultiply();
                        *dst = image::Rgb([un_pre.red(), un_pre.green(), un_pre.blue()]);
                    }
                }
                image.pixels = DynamicImage::ImageRgb8(rgb);
            } else {
                let mut rgba = RgbaImage::new(w, h);
                for (i, ((res, orig_px), dst)) in pixmap
                    .pixels()
                    .iter()
                    .zip(orig.pixels().iter())
                    .zip(rgba.pixels_mut().iter_mut())
                    .enumerate()
                {
                    if same_pixel(*res, precision.base[i]) {
                        *dst = image::Rgba([orig_px[0], orig_px[1], orig_px[2], 255]);
                    } else {
                        let un_pre = res.demultiply();
                        *dst = image::Rgba([
                            un_pre.red(),
                            un_pre.green(),
                            un_pre.blue(),
                            un_pre.alpha(),
                        ]);
                    }
                }
                image.pixels = DynamicImage::ImageRgba8(rgba);
            }
        }
        DynamicImage::ImageRgba8(orig) => {
            let mut rgba = RgbaImage::new(w, h);
            for (i, ((res, orig_px), dst)) in pixmap
                .pixels()
                .iter()
                .zip(orig.pixels().iter())
                .zip(rgba.pixels_mut().iter_mut())
                .enumerate()
            {
                if same_pixel(*res, precision.base[i]) {
                    *dst = *orig_px;
                } else {
                    let un_pre = res.demultiply();
                    *dst =
                        image::Rgba([un_pre.red(), un_pre.green(), un_pre.blue(), un_pre.alpha()]);
                }
            }
            image.pixels = DynamicImage::ImageRgba8(rgba);
        }
        DynamicImage::ImageLuma16(orig) => {
            let mut can_stay_luma = true;
            let mut can_stay_rgb = true;
            for (i, res) in pixmap.pixels().iter().enumerate() {
                if !same_pixel(*res, precision.base[i]) {
                    let un_pre = res.demultiply();
                    if un_pre.alpha() != 255 {
                        can_stay_luma = false;
                        can_stay_rgb = false;
                        break;
                    }
                    if un_pre.red() != un_pre.green() || un_pre.green() != un_pre.blue() {
                        can_stay_luma = false;
                    }
                }
            }
            if can_stay_luma {
                let mut out = ImageBuffer::<image::Luma<u16>, Vec<u16>>::new(w, h);
                for (i, ((res, orig_px), dst)) in pixmap
                    .pixels()
                    .iter()
                    .zip(orig.pixels().iter())
                    .zip(out.pixels_mut().iter_mut())
                    .enumerate()
                {
                    if same_pixel(*res, precision.base[i]) {
                        *dst = *orig_px;
                    } else {
                        let un_pre = res.demultiply();
                        *dst = image::Luma([un_pre.red() as u16 * 257]);
                    }
                }
                image.pixels = DynamicImage::ImageLuma16(out);
            } else if can_stay_rgb {
                let mut rgb = ImageBuffer::<image::Rgb<u16>, Vec<u16>>::new(w, h);
                for (i, ((res, orig_px), dst)) in pixmap
                    .pixels()
                    .iter()
                    .zip(orig.pixels().iter())
                    .zip(rgb.pixels_mut().iter_mut())
                    .enumerate()
                {
                    if same_pixel(*res, precision.base[i]) {
                        *dst = image::Rgb([orig_px[0], orig_px[0], orig_px[0]]);
                    } else {
                        let un_pre = res.demultiply();
                        *dst = image::Rgb([
                            un_pre.red() as u16 * 257,
                            un_pre.green() as u16 * 257,
                            un_pre.blue() as u16 * 257,
                        ]);
                    }
                }
                image.pixels = DynamicImage::ImageRgb16(rgb);
            } else {
                let mut rgba = ImageBuffer::<image::Rgba<u16>, Vec<u16>>::new(w, h);
                for (i, ((res, orig_px), dst)) in pixmap
                    .pixels()
                    .iter()
                    .zip(orig.pixels().iter())
                    .zip(rgba.pixels_mut().iter_mut())
                    .enumerate()
                {
                    if same_pixel(*res, precision.base[i]) {
                        *dst = image::Rgba([orig_px[0], orig_px[0], orig_px[0], u16::MAX]);
                    } else {
                        let un_pre = res.demultiply();
                        *dst = image::Rgba([
                            un_pre.red() as u16 * 257,
                            un_pre.green() as u16 * 257,
                            un_pre.blue() as u16 * 257,
                            un_pre.alpha() as u16 * 257,
                        ]);
                    }
                }
                image.pixels = DynamicImage::ImageRgba16(rgba);
            }
        }
        DynamicImage::ImageLumaA16(orig) => {
            let mut can_stay_luma = true;
            for (i, res) in pixmap.pixels().iter().enumerate() {
                if !same_pixel(*res, precision.base[i]) {
                    let un_pre = res.demultiply();
                    if un_pre.red() != un_pre.green() || un_pre.green() != un_pre.blue() {
                        can_stay_luma = false;
                        break;
                    }
                }
            }
            if can_stay_luma {
                let mut out = ImageBuffer::<image::LumaA<u16>, Vec<u16>>::new(w, h);
                for (i, ((res, orig_px), dst)) in pixmap
                    .pixels()
                    .iter()
                    .zip(orig.pixels().iter())
                    .zip(out.pixels_mut().iter_mut())
                    .enumerate()
                {
                    if same_pixel(*res, precision.base[i]) {
                        *dst = *orig_px;
                    } else {
                        let un_pre = res.demultiply();
                        *dst =
                            image::LumaA([un_pre.red() as u16 * 257, un_pre.alpha() as u16 * 257]);
                    }
                }
                image.pixels = DynamicImage::ImageLumaA16(out);
            } else {
                let mut rgba = ImageBuffer::<image::Rgba<u16>, Vec<u16>>::new(w, h);
                for (i, ((res, orig_px), dst)) in pixmap
                    .pixels()
                    .iter()
                    .zip(orig.pixels().iter())
                    .zip(rgba.pixels_mut().iter_mut())
                    .enumerate()
                {
                    if same_pixel(*res, precision.base[i]) {
                        *dst = image::Rgba([orig_px[0], orig_px[0], orig_px[0], orig_px[1]]);
                    } else {
                        let un_pre = res.demultiply();
                        *dst = image::Rgba([
                            un_pre.red() as u16 * 257,
                            un_pre.green() as u16 * 257,
                            un_pre.blue() as u16 * 257,
                            un_pre.alpha() as u16 * 257,
                        ]);
                    }
                }
                image.pixels = DynamicImage::ImageRgba16(rgba);
            }
        }
        DynamicImage::ImageRgb16(orig) => {
            let mut opaque = true;
            for (i, res) in pixmap.pixels().iter().enumerate() {
                if !same_pixel(*res, precision.base[i])
                    && res.alpha() != 255 {
                        opaque = false;
                        break;
                    }
            }
            if opaque {
                let mut rgb = ImageBuffer::<image::Rgb<u16>, Vec<u16>>::new(w, h);
                for (i, ((res, orig_px), dst)) in pixmap
                    .pixels()
                    .iter()
                    .zip(orig.pixels().iter())
                    .zip(rgb.pixels_mut().iter_mut())
                    .enumerate()
                {
                    if same_pixel(*res, precision.base[i]) {
                        *dst = *orig_px;
                    } else {
                        let un_pre = res.demultiply();
                        *dst = image::Rgb([
                            un_pre.red() as u16 * 257,
                            un_pre.green() as u16 * 257,
                            un_pre.blue() as u16 * 257,
                        ]);
                    }
                }
                image.pixels = DynamicImage::ImageRgb16(rgb);
            } else {
                let mut rgba = ImageBuffer::<image::Rgba<u16>, Vec<u16>>::new(w, h);
                for (i, ((res, orig_px), dst)) in pixmap
                    .pixels()
                    .iter()
                    .zip(orig.pixels().iter())
                    .zip(rgba.pixels_mut().iter_mut())
                    .enumerate()
                {
                    if same_pixel(*res, precision.base[i]) {
                        *dst = image::Rgba([orig_px[0], orig_px[1], orig_px[2], u16::MAX]);
                    } else {
                        let un_pre = res.demultiply();
                        *dst = image::Rgba([
                            un_pre.red() as u16 * 257,
                            un_pre.green() as u16 * 257,
                            un_pre.blue() as u16 * 257,
                            un_pre.alpha() as u16 * 257,
                        ]);
                    }
                }
                image.pixels = DynamicImage::ImageRgba16(rgba);
            }
        }
        DynamicImage::ImageRgba16(orig) => {
            let mut rgba = ImageBuffer::<image::Rgba<u16>, Vec<u16>>::new(w, h);
            for (i, ((res, orig_px), dst)) in pixmap
                .pixels()
                .iter()
                .zip(orig.pixels().iter())
                .zip(rgba.pixels_mut().iter_mut())
                .enumerate()
            {
                if same_pixel(*res, precision.base[i]) {
                    *dst = *orig_px;
                } else {
                    let un_pre = res.demultiply();
                    *dst = image::Rgba([
                        un_pre.red() as u16 * 257,
                        un_pre.green() as u16 * 257,
                        un_pre.blue() as u16 * 257,
                        un_pre.alpha() as u16 * 257,
                    ]);
                }
            }
            image.pixels = DynamicImage::ImageRgba16(rgba);
        }
        DynamicImage::ImageRgb32F(orig) => {
            let mut opaque = true;
            for (i, res) in pixmap.pixels().iter().enumerate() {
                if !same_pixel(*res, precision.base[i])
                    && res.alpha() != 255 {
                        opaque = false;
                        break;
                    }
            }
            if opaque {
                let mut rgb = ImageBuffer::<image::Rgb<f32>, Vec<f32>>::new(w, h);
                for (i, ((res, orig_px), dst)) in pixmap
                    .pixels()
                    .iter()
                    .zip(orig.pixels().iter())
                    .zip(rgb.pixels_mut().iter_mut())
                    .enumerate()
                {
                    if same_pixel(*res, precision.base[i]) {
                        *dst = *orig_px;
                    } else {
                        let un_pre = res.demultiply();
                        *dst = image::Rgb([
                            un_pre.red() as f32 / 255.0,
                            un_pre.green() as f32 / 255.0,
                            un_pre.blue() as f32 / 255.0,
                        ]);
                    }
                }
                image.pixels = DynamicImage::ImageRgb32F(rgb);
            } else {
                let mut rgba = ImageBuffer::<image::Rgba<f32>, Vec<f32>>::new(w, h);
                for (i, ((res, orig_px), dst)) in pixmap
                    .pixels()
                    .iter()
                    .zip(orig.pixels().iter())
                    .zip(rgba.pixels_mut().iter_mut())
                    .enumerate()
                {
                    if same_pixel(*res, precision.base[i]) {
                        *dst = image::Rgba([orig_px[0], orig_px[1], orig_px[2], 1.0]);
                    } else {
                        let un_pre = res.demultiply();
                        *dst = image::Rgba([
                            un_pre.red() as f32 / 255.0,
                            un_pre.green() as f32 / 255.0,
                            un_pre.blue() as f32 / 255.0,
                            un_pre.alpha() as f32 / 255.0,
                        ]);
                    }
                }
                image.pixels = DynamicImage::ImageRgba32F(rgba);
            }
        }
        DynamicImage::ImageRgba32F(orig) => {
            let mut rgba = ImageBuffer::<image::Rgba<f32>, Vec<f32>>::new(w, h);
            for (i, ((res, orig_px), dst)) in pixmap
                .pixels()
                .iter()
                .zip(orig.pixels().iter())
                .zip(rgba.pixels_mut().iter_mut())
                .enumerate()
            {
                if same_pixel(*res, precision.base[i]) {
                    *dst = *orig_px;
                } else {
                    let un_pre = res.demultiply();
                    *dst = image::Rgba([
                        un_pre.red() as f32 / 255.0,
                        un_pre.green() as f32 / 255.0,
                        un_pre.blue() as f32 / 255.0,
                        un_pre.alpha() as f32 / 255.0,
                    ]);
                }
            }
            image.pixels = DynamicImage::ImageRgba32F(rgba);
        }
        _ => {
            let mut out_rgba = RgbaImage::new(w, h);
            for (src, dst) in pixmap.pixels().iter().zip(out_rgba.pixels_mut().iter_mut()) {
                let un_pre = src.demultiply();
                *dst = image::Rgba([un_pre.red(), un_pre.green(), un_pre.blue(), un_pre.alpha()]);
            }
            image.pixels = DynamicImage::ImageRgba8(out_rgba);
        }
    }
}

/// Build a blur mask pixmap for a content region of the given size.
///
/// When `falloff` is `0.0` the mask is a solid white rectangle padded by
/// `inner_margin` on each side (sharp cutoff at the edges).
///
/// When `falloff` is positive the mask has the same solid core but an
/// additional gaussian-faded zone of that many pixels outside the inner
/// margin, so the blurred region graduates smoothly into the unblurred
/// background.
///
/// Returns `(mask_pixmap, total_margin)` where `total_margin` is the distance
/// from the content edge to the outer mask edge (equals `inner_margin` for
/// sharp, `inner_margin + falloff` for gradual).
fn build_blur_mask(
    content_w: f32,
    content_h: f32,
    inner_margin: f32,
    falloff: f32,
) -> Result<(Pixmap, f32), MagickError> {
    if falloff <= 0.0 {
        // Sharp-edged solid mask
        let mask_w = layer_dim(content_w + inner_margin * 2.0, "blur mask")?;
        let mask_h = layer_dim(content_h + inner_margin * 2.0, "blur mask")?;
        let mut mask = new_pixmap(mask_w, mask_h, "blur mask")?;
        for px in mask.pixels_mut() {
            *px = PremultipliedColorU8::from_rgba(255, 255, 255, 255).unwrap();
        }
        Ok((mask, inner_margin))
    } else {
        // Graduated edges: solid inner core + gaussian falloff fringe
        let total_margin = inner_margin + falloff;

        let mask_w = layer_dim(content_w + total_margin * 2.0, "gradual blur mask")?;
        let mask_h = layer_dim(content_h + total_margin * 2.0, "gradual blur mask")?;
        let mut mask = new_pixmap(mask_w, mask_h, "gradual blur mask")?;

        // Build a mask with gaussian-faded edges:
        // - Inside (content bounds + inner_margin): alpha = 255
        // - Outside that: alpha = 255 * exp(-dist² / (2 * falloff_sigma²))
        let falloff_sigma = falloff / 3.0; // so exp(-0.5 * (3σ/σ)²) ≈ 0
        let inv_2sig2 = 1.0 / (2.0 * falloff_sigma * falloff_sigma);

        // The inner rectangle within the mask where alpha stays 255
        let inner_x0 = total_margin - inner_margin;
        let inner_y0 = total_margin - inner_margin;
        let inner_x1 = inner_x0 + content_w + inner_margin * 2.0;
        let inner_y1 = inner_y0 + content_h + inner_margin * 2.0;

        let pixels = mask.pixels_mut();
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
                    pixels[idx] =
                        PremultipliedColorU8::from_rgba(alpha, alpha, alpha, alpha).unwrap();
                }
            }
        }
        Ok((mask, total_margin))
    }
}

/// Blur the region of `main_pixmap` that falls under a rotated mask.
///
/// `mask_src` is the local mask pixmap (e.g. the size of the QR or text layer).
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
) -> Result<(), MagickError> {
    if sigma <= 0.0 || sigma.is_nan() {
        return Ok(());
    }

    // Draw the mask through the rotation transform to get the rotated footprint
    // in canvas space.
    let mut mask_canvas = new_pixmap(img_w, img_h, "blur mask canvas")?;
    mask_canvas.draw_pixmap(
        0,
        0,
        mask_src.as_ref(),
        &PixmapPaint {
            opacity: 1.0,
            blend_mode: BlendMode::SourceOver,
            quality: FilterQuality::Bilinear,
        },
        transform,
        None,
    );

    // The footprint may be entirely off-canvas, in which case there is nothing
    // to blur and nothing to allocate.
    let Some((fx0, fx1, fy0, fy1)) = compute_alpha_bbox(mask_canvas.pixels(), img_w, img_h) else {
        return Ok(());
    };

    // Blur only the footprint's bounding box rather than the whole image. The
    // margin is the kernel's own support (`image`'s gaussian truncates at 3
    // sigma; 4 is used here for headroom) so every pixel the mask can reach
    // sees a full kernel and the result inside the mask is identical to
    // blurring the entire canvas.
    let radius = (sigma * 4.0).ceil() as i64 + 2;
    let x0 = (fx0 as i64 - radius).max(0) as u32;
    let y0 = (fy0 as i64 - radius).max(0) as u32;
    let x1 = (fx1 as i64 + radius).min(img_w as i64 - 1) as u32;
    let y1 = (fy1 as i64 + radius).min(img_h as i64 - 1) as u32;
    let crop_w = x1 - x0 + 1;
    let crop_h = y1 - y0 + 1;

    // Copy the sub-rect out row by row rather than going through a crop helper,
    // whose signature differs between `image` versions.
    // Premultiply the crop before blurring so transparent pixels' straight RGB
    // (usually black) does not bleed into the blurred fringe.
    let src_rgba = source_image.pixels.to_rgba8();
    let src_raw = src_rgba.as_raw();
    let src_stride = src_rgba.width() as usize * 4;
    let crop_stride = crop_w as usize * 4;
    let mut crop_raw = vec![0u8; crop_stride * crop_h as usize];
    for cy in 0..crop_h as usize {
        let s = (y0 as usize + cy) * src_stride + x0 as usize * 4;
        let d = cy * crop_stride;
        for cx in 0..crop_w as usize {
            let sp = s + cx * 4;
            let dp = d + cx * 4;
            let a = src_raw[sp + 3] as u32;
            crop_raw[dp] = ((src_raw[sp] as u32 * a) / 255) as u8;
            crop_raw[dp + 1] = ((src_raw[sp + 1] as u32 * a) / 255) as u8;
            crop_raw[dp + 2] = ((src_raw[sp + 2] as u32 * a) / 255) as u8;
            crop_raw[dp + 3] = a as u8;
        }
    }
    let crop = RgbaImage::from_raw(crop_w, crop_h, crop_raw)
        .ok_or_else(|| wm_err!("failed to build blur crop buffer"))?;
    let blurred = DynamicImage::ImageRgba8(crop).blur(sigma);
    let blurred_raw = blurred.to_rgba8().into_raw();

    // Blend: where the mask has coverage, lerp the blurred pixels into the
    // canvas.
    //
    // This is done explicitly on premultiplied bytes rather than by handing the
    // mask to `draw_pixmap`. tiny_skia folds clip-mask coverage into the source
    // alpha, so a masked `Source` blend punches the canvas transparent wherever
    // coverage is partial or zero - which showed up as dark borders around
    // every blurred region and around the whole crop rectangle. The lerp below
    // touches only pixels with non-zero coverage, and because each premultiplied
    // colour channel is <= its alpha on both inputs and integer division is
    // monotonic, the result always satisfies that invariant too; `from_rgba`
    // never rejects a pixel here.
    let w = img_w as usize;
    let bx0 = x0 as usize;
    let by0 = y0 as usize;
    let bw = crop_w as usize;
    let bh = crop_h as usize;
    let mask_pixels = mask_canvas.pixels();

    main_pixmap.pixels_mut().par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        if y < by0 || y >= by0 + bh {
            return;
        }
        let mask_row = y * w;
        let blur_row = (y - by0) * bw;
        for x in bx0..bx0 + bw {
            let ma = mask_pixels[mask_row + x].alpha() as u32;
            if ma == 0 {
                continue;
            }
            let bi = (blur_row + (x - bx0)) * 4;
            let ba = blurred_raw[bi + 3];
            let br = blurred_raw[bi].min(ba);
            let bg = blurred_raw[bi + 1].min(ba);
            let bb = blurred_raw[bi + 2].min(ba);
            let blurred_pre = PremultipliedColorU8::from_rgba(br, bg, bb, ba)
                .unwrap_or_else(|| PremultipliedColorU8::from_rgba(0, 0, 0, 0).unwrap());

            let orig = row[x];
            let inv = 255 - ma;
            let out_r = ((blurred_pre.red() as u32 * ma + orig.red() as u32 * inv) / 255) as u8;
            let out_g = ((blurred_pre.green() as u32 * ma + orig.green() as u32 * inv) / 255) as u8;
            let out_b = ((blurred_pre.blue() as u32 * ma + orig.blue() as u32 * inv) / 255) as u8;
            let out_a = ((blurred_pre.alpha() as u32 * ma + orig.alpha() as u32 * inv) / 255) as u8;

            if let Some(c) = PremultipliedColorU8::from_rgba(out_r, out_g, out_b, out_a) {
                row[x] = c;
            }
        }
    });

    Ok(())
}

/// One-dimensional grayscale max filter using the van Herk / Gil-Werman
/// algorithm: three linear passes regardless of the window size, so the cost is
/// O(n) rather than O(n * radius).
///
/// Computes `dst[i] = max(src[i - radius ..= i + radius])`, treating
/// out-of-range samples as 0 (the identity for a max, i.e. zero padding).
/// `dst` must be at least as long as `src`.
fn max_filter_1d(src: &[u8], dst: &mut [u8], radius: usize) {
    let n = src.len();
    if n == 0 {
        return;
    }
    if radius == 0 {
        dst[..n].copy_from_slice(src);
        return;
    }

    let k = 2 * radius + 1;
    // Pad by `radius` on the left and enough on the right that every output
    // window is covered and the total length is a whole number of blocks.
    let padded_len = (n + 2 * radius).div_ceil(k) * k;
    let mut padded = vec![0u8; padded_len];
    padded[radius..radius + n].copy_from_slice(src);

    // Running maxima forwards and backwards within each block of `k` samples.
    let mut prefix = vec![0u8; padded_len];
    let mut suffix = vec![0u8; padded_len];
    let mut block = 0;
    while block < padded_len {
        let end = block + k;
        let mut m = 0u8;
        for j in block..end {
            m = m.max(padded[j]);
            prefix[j] = m;
        }
        let mut m = 0u8;
        for j in (block..end).rev() {
            m = m.max(padded[j]);
            suffix[j] = m;
        }
        block = end;
    }

    // A window of exactly `k` samples spans at most two blocks, so the suffix
    // at its start and the prefix at its end cover it exactly.
    for x in 0..n {
        dst[x] = prefix[x + k - 1].max(suffix[x]);
    }
}

/// Cache-blocked transpose. The dilation below transposes once per rectangle,
/// so the tiling is worth roughly a 2x saving on that part of the work.
fn transpose(src: &[u8], width: usize, height: usize) -> Vec<u8> {
    const BLOCK: usize = 32;
    let mut out = vec![0u8; src.len()];
    let mut y0 = 0;
    while y0 < height {
        let y_end = (y0 + BLOCK).min(height);
        let mut x0 = 0;
        while x0 < width {
            let x_end = (x0 + BLOCK).min(width);
            for y in y0..y_end {
                for x in x0..x_end {
                    out[x * height + y] = src[y * width + x];
                }
            }
            x0 += BLOCK;
        }
        y0 += BLOCK;
    }
    out
}

/// Run the 1-D max filter across every row, in parallel.
fn max_filter_rows(buf: &mut [u8], width: usize, radius: usize) {
    if radius == 0 || width == 0 {
        return;
    }
    buf.par_chunks_mut(width).for_each(|row| {
        let src = row.to_vec();
        max_filter_1d(&src, row, radius);
    });
}

/// Integer square root, computed exactly so the structuring element can't drift
/// by a pixel from the `dx*dx + dy*dy <= radius*radius` test it must reproduce.
fn isqrt(v: usize) -> usize {
    let mut x = (v as f64).sqrt() as usize;
    while x > 0 && x * x > v {
        x -= 1;
    }
    while (x + 1) * (x + 1) <= v {
        x += 1;
    }
    x
}

/// Decompose the digital disc of radius `r` into the minimal set of nested
/// rectangles whose union is exactly the disc.
///
/// Row `dy` of the disc is the segment `|dx| <= c(dy)` with
/// `c(dy) = isqrt(r^2 - dy^2)`, and `c` is non-increasing. So the disc is the
/// union of `[-c(j), c(j)] x [-j, j]` over all `j`, and a rectangle is redundant
/// unless its chord length is the last of its value - which leaves one
/// rectangle per distinct chord length, about 0.6 * r of them.
fn disc_rectangles(r: usize) -> Vec<(usize, usize)> {
    let mut rects = Vec::new();
    let mut last_chord = usize::MAX;
    // Scanning `j` downwards makes the first `j` at which a chord length
    // appears the largest one having it, which is the rectangle to keep.
    for j in (0..=r).rev() {
        let chord = isqrt(r * r - j * j);
        if chord != last_chord {
            rects.push((chord, j));
            last_chord = chord;
        }
    }
    rects
}

/// Morphological dilation on an alpha channel using a circular structuring
/// element. `thickness` is the outline stroke width in pixels.
///
/// The structuring element is the exact digital disc `dx^2 + dy^2 <= radius^2`,
/// so the output is byte-for-byte what direct evaluation of the k^2 kernel
/// produced - no octagonal or otherwise faceted approximation. It gets there by
/// taking the union of the nested rectangles that tile the disc, each rectangle
/// being one horizontal and one vertical 1-D max filter. Every such filter is
/// O(w*h) via van Herk / Gil-Werman, so the whole dilation is O(w*h*radius)
/// instead of O(w*h*radius^2): about 10x faster at thickness 15, 30x at 31 and
/// 60x at 65, measured on a 1500x800 layer.
///
/// Two details keep it exact and cheap. Horizontal radii are applied
/// incrementally, since composing max filters adds their radii
/// (`H_a . H_b == H_{a+b}`), and the vertical pass runs in transposed space so
/// one transpose per rectangle suffices instead of two. Border handling needs no
/// padding buffer here: within a single rectangle the horizontal pass only ever
/// reads its own row and the vertical pass only its own column, so nothing that
/// spills off an edge could have come back.
fn dilate_alpha(alpha: &[u8], width: u32, height: u32, thickness: u32) -> Vec<u8> {
    let radius = thickness as usize;

    let w = width as usize;
    let h = height as usize;
    if radius == 0 || w == 0 || h == 0 || alpha.len() < w * h {
        return alpha.to_vec();
    }

    let mut rects = disc_rectangles(radius);
    rects.sort_by_key(|&(chord, _)| chord);

    let mut hbuf = alpha[..w * h].to_vec();
    // Accumulate in transposed orientation so the vertical pass is a row pass.
    let mut acc_t = vec![0u8; w * h];
    let mut applied_chord = 0usize;

    for (chord, half_height) in rects {
        max_filter_rows(&mut hbuf, w, chord - applied_chord);
        applied_chord = chord;

        let mut t = transpose(&hbuf, w, h);
        max_filter_rows(&mut t, h, half_height);
        acc_t.par_iter_mut().zip(t.par_iter()).for_each(|(acc, v)| {
            if *v > *acc {
                *acc = *v;
            }
        });
    }

    transpose(&acc_t, h, w)
}

/// Apply radial explode displacement to a pixmap in-place.
fn apply_explode(
    pixmap: &mut Pixmap,
    cx: f32,
    cy: f32,
    effect_radius: f32,
    strength: f32,
    img_w: u32,
    img_h: u32,
) {
    if strength <= 0.0 || effect_radius <= 0.0 {
        return;
    }
    let inv_radius = 1.0 / effect_radius;
    let src_snap: Vec<PremultipliedColorU8> = pixmap.pixels().to_vec();
    let w = img_w as usize;
    let h = img_h as usize;

    pixmap.pixels_mut().par_chunks_mut(w).enumerate().for_each(|(y_idx, row)| {
        for x_idx in 0..w {
            let dx = x_idx as f32 - cx;
            let dy = y_idx as f32 - cy;
            let dist = (dx * dx + dy * dy).sqrt();

            if dist >= effect_radius || dist < 0.001 {
                continue;
            }

            // Quadratic falloff: strong near center, zero at radius
            let t = dist * inv_radius;
            let displacement = strength * (1.0 - t) * (1.0 - t);

            // Pull source position back toward centroid
            let scale = ((dist - displacement) / dist).max(0.0);
            let src_x = (cx + dx * scale).clamp(0.0, (w - 1) as f32);
            let src_y = (cy + dy * scale).clamp(0.0, (h - 1) as f32);

            let x0 = src_x.floor() as usize;
            let y0 = src_y.floor() as usize;
            let x1 = (x0 + 1).min(w - 1);
            let y1 = (y0 + 1).min(h - 1);
            let fx = src_x - x0 as f32;
            let fy = src_y - y0 as f32;

            let p00 = src_snap[y0 * w + x0];
            let p10 = src_snap[y0 * w + x1];
            let p01 = src_snap[y1 * w + x0];
            let p11 = src_snap[y1 * w + x1];

            let blerp = |a: u8, b: u8, c: u8, d: u8| -> u8 {
                let top = a as f32 * (1.0 - fx) + b as f32 * fx;
                let bot = c as f32 * (1.0 - fx) + d as f32 * fx;
                (top * (1.0 - fy) + bot * fy) as u8
            };

            let r = blerp(p00.red(), p10.red(), p01.red(), p11.red());
            let g = blerp(p00.green(), p10.green(), p01.green(), p11.green());
            let b = blerp(p00.blue(), p10.blue(), p01.blue(), p11.blue());
            let a = blerp(p00.alpha(), p10.alpha(), p01.alpha(), p11.alpha());

            if let Some(c) = PremultipliedColorU8::from_rgba(r, g, b, a) {
                row[x_idx] = c;
            }
        }
    });
}

/// Apply text-shaped meltdown displacement to a pixmap in-place.
fn apply_meltdown(
    pixmap: &mut Pixmap,
    field: &[f32],
    is_omni: bool,
    fixed_dx: f32,
    fixed_dy: f32,
    strength: f32,
    img_w: u32,
    img_h: u32,
) {
    if strength <= 0.0 {
        return;
    }
    let src_snap: Vec<PremultipliedColorU8> = pixmap.pixels().to_vec();
    let w = img_w as usize;
    let h = img_h as usize;

    pixmap.pixels_mut().par_chunks_mut(w).enumerate().for_each(|(y_idx, row)| {
        for x_idx in 0..w {
            let fv = field[y_idx * w + x_idx];
            if fv < 0.002 {
                continue;
            }

            let displacement = strength * fv;

            // Determine push direction
            let (push_dx, push_dy) = if is_omni {
                let x0 = x_idx.saturating_sub(1);
                let x1 = (x_idx + 1).min(w - 1);
                let y0 = y_idx.saturating_sub(1);
                let y1 = (y_idx + 1).min(h - 1);
                let gx = field[y_idx * w + x1] - field[y_idx * w + x0];
                let gy = field[y1 * w + x_idx] - field[y0 * w + x_idx];
                let glen = (gx * gx + gy * gy).sqrt();
                if glen < 1e-6 {
                    continue;
                }
                (-gx / glen, -gy / glen)
            } else {
                (fixed_dx, fixed_dy)
            };

            let src_x = (x_idx as f32 - displacement * push_dx).clamp(0.0, (w - 1) as f32);
            let src_y = (y_idx as f32 - displacement * push_dy).clamp(0.0, (h - 1) as f32);

            let sx0 = src_x.floor() as usize;
            let sy0 = src_y.floor() as usize;
            let sx1 = (sx0 + 1).min(w - 1);
            let sy1 = (sy0 + 1).min(h - 1);
            let fx = src_x - sx0 as f32;
            let fy = src_y - sy0 as f32;

            let p00 = src_snap[sy0 * w + sx0];
            let p10 = src_snap[sy0 * w + sx1];
            let p01 = src_snap[sy1 * w + sx0];
            let p11 = src_snap[sy1 * w + sx1];

            let blerp = |a: u8, b: u8, c: u8, d: u8| -> u8 {
                let top = a as f32 * (1.0 - fx) + b as f32 * fx;
                let bot = c as f32 * (1.0 - fx) + d as f32 * fx;
                (top * (1.0 - fy) + bot * fy) as u8
            };

            let r = blerp(p00.red(), p10.red(), p01.red(), p11.red());
            let g = blerp(p00.green(), p10.green(), p01.green(), p11.green());
            let b = blerp(p00.blue(), p10.blue(), p01.blue(), p11.blue());
            let a = blerp(p00.alpha(), p10.alpha(), p01.alpha(), p11.alpha());

            if let Some(c) = PremultipliedColorU8::from_rgba(r, g, b, a) {
                row[x_idx] = c;
            }
        }
    });
}

/// Render a QR code block onto the image, using the same position/rotation/color
/// system as text rendering. The QR code size is determined by font_size.
/// Renders at 4x internal resolution and downsamples with area-averaging for smooth rotated edges.
fn render_qr_block(
    image: &mut Image,
    config: &TextConfig,
    qr: &QrBlock,
) -> Result<(), MagickError> {
    // The QR path only understands the QR block's own blur field. Anything set
    // in the effect column would otherwise be dropped without a word.
    if !matches!(config.effect, TextEffect::None)
        || !matches!(config.displacement, DisplacementEffect::None)
    {
        eprintln!(
            "warning: text effects and displacement modifiers are not applied to QR blocks; \
             use the QR block's own blur field (e.g. '{{QR:H:100+15:#FFFFFFCC:...}}') instead"
        );
    }

    let (mut main_pixmap, precision) = begin_render(image)?;
    let img_w = main_pixmap.width();
    let img_h = main_pixmap.height();

    let qr_size = match config.font_size {
        FontSize::Absolute(v) => v.max(1.0),
        FontSize::RelativePercent(pct) => (img_h as f32 * (pct / 100.0)).max(1.0),
    };
    if !qr_size.is_finite() || qr_size > 20_000.0 {
        return Err(wm_err!("QR size out of range: {}", qr_size));
    }

    let code = QrCode::with_error_correction_level(qr.content.as_bytes(), qr.ec_level)
        .map_err(|e| wm_err!("QR generation failed: {}", e))?;

    let (r, g, b, a) = config.color;
    let (lr, lg, lb, la) = qr.light_color;

    // Render QR at 4x supersampled resolution for smooth rotated edges
    let ss = 4u32;
    let ss_size = ((qr_size * ss as f32).ceil() as u32).max(1);
    let qr_rgba: RgbaImage = code
        .render::<image::Rgba<u8>>()
        .dark_color(image::Rgba([r, g, b, a]))
        .light_color(image::Rgba([lr, lg, lb, la]))
        .min_dimensions(ss_size, ss_size)
        .build();

    let (ss_w, ss_h) = qr_rgba.dimensions();
    let qw = ss_w as f32 / ss as f32;
    let qh = ss_h as f32 / ss as f32;

    // Build a tiny_skia pixmap from the supersampled QR image
    let mut qr_pixmap = new_pixmap(ss_w, ss_h, "QR")?;
    for (src, dst) in qr_rgba.pixels().iter().zip(qr_pixmap.pixels_mut().iter_mut()) {
        *dst = ColorU8::from_rgba(src[0], src[1], src[2], src[3]).premultiply();
    }

    // Compute the bounding box of the QR after rotation so positioning
    // accounts for the full rotated extent (prevents corner clipping).
    let angle_rad = config.rotation.to_radians();
    let cos_a = angle_rad.cos().abs();
    let sin_a = angle_rad.sin().abs();
    let rot_w = qw * cos_a + qh * sin_a;
    let rot_h = qw * sin_a + qh * cos_a;

    // Resolve position using the *rotated* bounding box size so that e.g.
    // x=0 means the leftmost rotated corner sits at x=0.
    let desired_x = config.x.resolve(img_w as f32, rot_w, qr_size);
    let desired_y = config.y.resolve(img_h as f32, rot_h, qr_size);

    // The rotation pivot is at (qw/2, qh/2) relative to the translate origin.
    // After rotation the bounding box top-left shifts by (rot_w-qw)/2 in each axis.
    // Compensate so the *rotated* box lands at (desired_x, desired_y).
    let start_x = desired_x + (rot_w - qw) / 2.0;
    let start_y = desired_y + (rot_h - qh) / 2.0;

    // If blur_sigma > 0, blur the region under the *rotated* QR footprint.
    if qr.blur_sigma > 0.0 {
        // Build the mask at display resolution using shared blur mask builder.
        // Use a small inner margin so the blur extends just past the QR edges.
        // Both margins are proportional to the *rendered* size rather than the
        // requested one, so the frame stays in step with what is drawn.
        let inner_margin = qw * 0.05;
        // Compute the gradual falloff distance from the percentage of QR size.
        // 0% = sharp edges (default), e.g. 15% on a 200px QR → 30px falloff zone.
        let falloff =
            if qr.blur_gradual_pct > 0.0 { qw * (qr.blur_gradual_pct / 100.0) } else { 0.0 };
        let (mask_src, total_margin) = build_blur_mask(qw, qh, inner_margin, falloff)?;

        // Position the mask so it's centered on the QR display footprint.
        let mask_x = start_x - total_margin;
        let mask_y = start_y - total_margin;
        let mask_transform =
            Transform::from_translate(mask_x, mask_y).pre_concat(Transform::from_rotate_at(
                config.rotation,
                qw / 2.0 + total_margin,
                qh / 2.0 + total_margin,
            ));

        apply_blur_under_rotated_region(
            &mut main_pixmap,
            image,
            &mask_src,
            mask_transform,
            qr.blur_sigma,
            img_w,
            img_h,
        )?;
    }

    // To anti-alias the outside rotated edges as well as the interior QR modules,
    // draw the 4x QR onto a 4x supersampled local crop pixmap at 4x resolution,
    // then downsample the entire rotated crop with Lanczos3 area-averaging.
    let center_x = start_x + qw / 2.0;
    let center_y = start_y + qh / 2.0;
    let bx0 = (center_x - rot_w / 2.0).floor() as i64 - 1;
    let by0 = (center_y - rot_h / 2.0).floor() as i64 - 1;
    let bx1 = (center_x + rot_w / 2.0).ceil() as i64 + 1;
    let by1 = (center_y + rot_h / 2.0).ceil() as i64 + 1;
    let bw = (bx1 - bx0 + 1).max(1) as u32;
    let bh = (by1 - by0 + 1).max(1) as u32;

    let ss_crop_w = layer_dim((bw * ss) as f32, "QR SS crop")?;
    let ss_crop_h = layer_dim((bh * ss) as f32, "QR SS crop")?;
    let mut ss_crop_pixmap = new_pixmap(ss_crop_w, ss_crop_h, "QR SS crop")?;

    let crop_cx_4x = (center_x - bx0 as f32) * ss as f32;
    let crop_cy_4x = (center_y - by0 as f32) * ss as f32;

    let transform_4x =
        Transform::from_translate(crop_cx_4x - ss_w as f32 / 2.0, crop_cy_4x - ss_h as f32 / 2.0)
            .pre_concat(Transform::from_rotate_at(
                config.rotation,
                ss_w as f32 / 2.0,
                ss_h as f32 / 2.0,
            ));

    ss_crop_pixmap.draw_pixmap(
        0,
        0,
        qr_pixmap.as_ref(),
        &PixmapPaint {
            opacity: 1.0,
            blend_mode: BlendMode::SourceOver,
            quality: FilterQuality::Bilinear,
        },
        transform_4x,
        None,
    );

    let mut ss_img = RgbaImage::new(ss_crop_w, ss_crop_h);
    for (src, dst) in ss_crop_pixmap.pixels().iter().zip(ss_img.pixels_mut().iter_mut()) {
        let un_pre = src.demultiply();
        *dst = image::Rgba([un_pre.red(), un_pre.green(), un_pre.blue(), un_pre.alpha()]);
    }

    let downscaled =
        image::imageops::resize(&ss_img, bw, bh, image::imageops::FilterType::Lanczos3);

    let mut crop_1x_pixmap = new_pixmap(bw, bh, "QR downscaled")?;
    for (src, dst) in downscaled.pixels().iter().zip(crop_1x_pixmap.pixels_mut().iter_mut()) {
        *dst = ColorU8::from_rgba(src[0], src[1], src[2], src[3]).premultiply();
    }

    let final_transform = Transform::from_translate(bx0 as f32, by0 as f32);
    main_pixmap.draw_pixmap(
        0,
        0,
        crop_1x_pixmap.as_ref(),
        &PixmapPaint {
            opacity: 1.0,
            blend_mode: BlendMode::SourceOver,
            quality: FilterQuality::Bilinear,
        },
        final_transform,
        None,
    );

    finish_render(image, &main_pixmap, precision);
    Ok(())
}

pub fn render_text(image: &mut Image, config: &TextConfig) -> Result<(), MagickError> {
    //eprintln!("text: input color type = {:?}", image.pixels.color());
    // Parse text for QR blocks and escape sequences. `TextConfig::parse_arg`
    // already validated this, so a failure here means the config was built by
    // hand rather than parsed.
    let segments = parse_text_segments(&config.text).map_err(|e| {
        wm_err!("{}", e.message.as_deref().unwrap_or("invalid QR block in text field"))
    })?;

    // If the entire text is a single QR block, render only the QR code
    if segments.len() == 1
        && let TextSegment::Qr(ref qr) = segments[0]
    {
        return render_qr_block(image, config, qr);
    }

    // For mixed content or plain text, render QR blocks first (each one composited),
    // then render the remaining plain text on top.
    for seg in &segments {
        if let TextSegment::Qr(qr) = seg {
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
    let text_config = TextConfig { text: plain_text, ..config.clone() };
    render_text_inner(image, &text_config)
}

fn render_text_inner(image: &mut Image, config: &TextConfig) -> Result<(), MagickError> {
    let (mut main_pixmap, precision) = begin_render(image)?;
    let img_w = main_pixmap.width();
    let img_h = main_pixmap.height();

    // Shared, process-wide font system and glyph cache. Recovering from a
    // poisoned mutex is safe here: the guarded state is a cache, and a panic
    // mid-render can't leave it logically inconsistent.
    let mut fonts = shared_fonts().lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let SharedFonts { font_system, swash_cache } = &mut *fonts;

    // Support colon-separated font names as fallback list (e.g. "Iosevka:Twemoji:sans-serif")
    let font_names: Vec<&str> = config.font_name.split(':').map(|s| s.trim()).collect();
    let mut available_fonts: Vec<&str> = Vec::new();
    for name in &font_names {
        let found = font_system.db().faces().any(|info| {
            info.families.iter().any(|(fname, _): &(String, _)| fname.eq_ignore_ascii_case(name))
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
        FontSize::Absolute(v) => v.max(1.0),
        FontSize::RelativePercent(pct) => (img_h as f32 * (pct / 100.0)).max(1.0),
    };

    // Rotated text needs a larger axis-aligned box than its own width and
    // height, so both the autofit test and the positioning below work against
    // the rotated extent.
    let rot_rad = config.rotation.to_radians();
    let cos_r = rot_rad.cos().abs();
    let sin_r = rot_rad.sin().abs();

    // Autofit: measure natural text size, shrink font if it exceeds image bounds
    let mut font_size = initial_font_size;
    let (actual_w, actual_h, line_height) = loop {
        let lh = font_size * 1.2;
        let metrics = Metrics::new(font_size, lh);
        let mut buf = Buffer::new(&mut *font_system, metrics);
        {
            let mut br = buf.borrow_with(&mut *font_system);
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

            // Nothing measurable to fit; also keeps the ratios below finite.
            if tw <= 0.0 || tw.is_nan() || th <= 0.0 || th.is_nan() {
                break (tw, th, lh);
            }

            let fit_w = tw * cos_r + th * sin_r;
            let fit_h = tw * sin_r + th * cos_r;

            if (fit_w <= img_w as f32 && fit_h <= img_h as f32) || font_size <= 1.0 {
                break (tw, th, lh);
            }

            let scale = (img_w as f32 / fit_w).min(img_h as f32 / fit_h);
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
    let mut buffer = Buffer::new(&mut *font_system, metrics);
    {
        let mut buffer_ref = buffer.borrow_with(&mut *font_system);
        let attrs = Attrs::new().family(Family::Name(primary_font));
        buffer_ref.set_text(&config.text, &attrs, Shaping::Advanced, Some(config.justify));
        // Use actual text width so justification (center/right) works correctly.
        // Rounded up by a fraction of a pixel so that the longest line, whose
        // width is exactly `actual_w`, can't re-wrap on a float comparison.
        buffer_ref.set_size(Some(actual_w.ceil().max(1.0)), None);
        buffer_ref.shape_until_scroll(true);
    }

    let tw_u32 = layer_dim(text_w, "text")?;
    let th_u32 = layer_dim(text_h, "text")?;

    let mut text_pixmap = new_pixmap(tw_u32, th_u32, "text")?;

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
            let physical_glyph = glyph.physical((pad, pad + run.line_y), 1.0);
            if let Some(img) = swash_cache.get_image(&mut *font_system, physical_glyph.cache_key) {
                // Offset by pad is included in physical_glyph coordinate.
                let gx = physical_glyph.x + img.placement.left;
                let gy = physical_glyph.y - img.placement.top;

                draw_glyph_pixels(&mut text_pixmap, tw_u32, th_u32, img, gx, gy, config.color);
            } else {
                // Save position and text for fallback font rendering.
                // `glyph.start`/`glyph.end` are byte offsets into the *layout
                // run's* text, not into the whole buffer, so slice `run.text`;
                // indexing `config.text` picked the wrong characters on every
                // line after the first. Snap to char boundaries since glyph
                // byte ranges may land mid-codepoint.
                let rt = run.text;
                let start = (0..=glyph.start.min(rt.len()))
                    .rev()
                    .find(|&i| rt.is_char_boundary(i))
                    .unwrap_or(0);
                let end = (glyph.end.min(rt.len())..=rt.len())
                    .find(|&i| rt.is_char_boundary(i))
                    .unwrap_or(rt.len())
                    .max(start);
                let ch = &rt[start..end];
                if !ch.is_empty() {
                    failed.push(FailedGlyph { text: ch.to_string(), x: glyph.x, y: run.line_y });
                }
            }
        }
    }

    // Attempt to render failed glyphs with fallback fonts
    if !failed.is_empty() && available_fonts.len() > 1 {
        for &fallback_name in &available_fonts[1..] {
            let mut still_failed = Vec::new();
            for fg in failed {
                let metrics = Metrics::new(font_size, line_height);
                let mut fb_buf = Buffer::new(&mut *font_system, metrics);
                {
                    let mut fb_br = fb_buf.borrow_with(&mut *font_system);
                    let attrs = Attrs::new().family(Family::Name(fallback_name));
                    fb_br.set_text(&fg.text, &attrs, Shaping::Advanced, None);
                    fb_br.set_size(None, None);
                    fb_br.shape_until_scroll(true);
                }

                let mut rendered = false;
                for run in fb_buf.layout_runs() {
                    for glyph in run.glyphs.iter() {
                        let pg = glyph.physical((pad + fg.x, pad + fg.y + run.line_y), 1.0);
                        if let Some(img) = swash_cache.get_image(&mut *font_system, pg.cache_key) {
                            rendered = true;
                            // Draw at the original position from primary font layout
                            let gx = pg.x + img.placement.left;
                            let gy = pg.y - img.placement.top;
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

    // Everything below is pure raster work; release the font lock so other
    // renders can shape while this one composites.
    drop(buffer);
    drop(fonts);

    // Position based on actual visible text bounds, then subtract pad
    // so the pixmap's padding margin extends outside the positioned area.
    // Resolve against the rotated extent so edge-anchored positions (0%, 100%,
    // absolute) don't push rotated text off the canvas; the pixmap rotates
    // about its own centre, which is also the centre of the visible text, so
    // the rotated box's top-left shifts by half the growth in each axis.
    let rot_w = actual_w * cos_r + actual_h * sin_r;
    let rot_h = actual_w * sin_r + actual_h * cos_r;
    let start_x = config.x.resolve(img_w as f32, rot_w, font_size) + (rot_w - actual_w) / 2.0 - pad;
    let start_y = config.y.resolve(img_h as f32, rot_h, font_size) + (rot_h - actual_h) / 2.0 - pad;

    let transform = Transform::from_translate(start_x, start_y)
        .pre_concat(Transform::from_rotate_at(config.rotation, text_w / 2.0, text_h / 2.0));

    let paint = PixmapPaint {
        opacity: 1.0,
        blend_mode: BlendMode::SourceOver,
        quality: FilterQuality::Bilinear,
    };

    // Separate background blur and glyph effects (outline/shadow), supporting combined effects.
    let (bg_effect, glyph_effect) = match &config.effect {
        TextEffect::None => (None, None),
        TextEffect::Blur { .. } | TextEffect::GradualBlur { .. } => (Some(&config.effect), None),
        TextEffect::Outline { .. } | TextEffect::Shadow { .. } => (None, Some(&config.effect)),
        TextEffect::Combined { bg, glyph } => (Some(&**bg), Some(&**glyph)),
    };

    // 1. Apply background blur effect (if any)
    if let Some(bg) = bg_effect
        && let TextEffect::Blur { sigma } | TextEffect::GradualBlur { sigma } = bg
            && let Some((min_x, max_x, min_y, max_y)) =
                compute_alpha_bbox(text_pixmap.pixels(), tw_u32, th_u32)
            {
                let glyph_w = (max_x - min_x + 1) as f32;
                let glyph_h = (max_y - min_y + 1) as f32;

                // Small symmetric margin around the tight glyph bounds
                let inner_margin = font_size * 0.3;
                // GradualBlur gets a smooth falloff zone; plain Blur stays sharp
                let falloff = if matches!(bg, TextEffect::GradualBlur { .. }) {
                    (*sigma * 3.0).max(font_size * 0.5)
                } else {
                    0.0
                };
                let (mask_src, total_margin) =
                    build_blur_mask(glyph_w, glyph_h, inner_margin, falloff)?;

                // The glyph bounding box top-left within the text pixmap is (min_x, min_y).
                // The text pixmap top-left on the canvas is (start_x, start_y).
                // So the glyph top-left on the canvas is (start_x + min_x, start_y + min_y).
                // The mask starts total_margin before that.
                let mask_x = start_x + min_x as f32 - total_margin;
                let mask_y = start_y + min_y as f32 - total_margin;

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
                )?;
            }

    // 2. Prepare glyph outline or shadow layer (if configured)
    let effect_layer = if let Some(glyph) = glyph_effect {
        match glyph {
            TextEffect::Outline { thickness, color } => {
                // Dilation grows the mask by up to thickness in every direction,
                // which the text pixmap's own `pad` margin doesn't necessarily
                // cover, so work in a buffer expanded by that much and composite it
                // back at the matching offset.
                let grow = *thickness + 1;
                let ow = tw_u32 + grow * 2;
                let oh = th_u32 + grow * 2;

                // Extract alpha channel from text_pixmap into the expanded buffer
                let mut alpha_buf = vec![0u8; ow as usize * oh as usize];
                for (i, px) in text_pixmap.pixels().iter().enumerate() {
                    let x = (i as u32) % tw_u32 + grow;
                    let y = (i as u32) / tw_u32 + grow;
                    alpha_buf[(y * ow + x) as usize] = px.alpha();
                }

                // Dilate the alpha channel
                let dilated = dilate_alpha(&alpha_buf, ow, oh, *thickness);

                // Build an outline pixmap: dilated area filled with outline color
                let mut outline_pixmap = new_pixmap(ow, oh, "outline")?;
                let (or, og, ob, oa) = *color;
                for (i, out_px) in outline_pixmap.pixels_mut().iter_mut().enumerate() {
                    let da = dilated[i] as u32;
                    if da > 0 {
                        // Modulate outline color alpha by the dilated mask
                        let final_a = (oa as u32 * da) / 255;
                        let pr = (or as u32 * final_a) / 255;
                        let pg = (og as u32 * final_a) / 255;
                        let pb = (ob as u32 * final_a) / 255;
                        if let Some(c) = PremultipliedColorU8::from_rgba(
                            pr as u8,
                            pg as u8,
                            pb as u8,
                            final_a as u8,
                        ) {
                            *out_px = c;
                        }
                    }
                }

                // Project outline to canvas space
                let outline_transform =
                    Transform::from_translate(start_x - grow as f32, start_y - grow as f32)
                        .pre_concat(Transform::from_rotate_at(
                            config.rotation,
                            text_w / 2.0 + grow as f32,
                            text_h / 2.0 + grow as f32,
                        ));
                let mut layer = new_pixmap(img_w, img_h, "outline layer")?;
                layer.draw_pixmap(0, 0, outline_pixmap.as_ref(), &paint, outline_transform, None);
                Some(layer)
            }
            TextEffect::Shadow { dx, dy, sigma, color } => {
                // Build a shadow image from the already-rendered text alpha,
                // filled with the shadow color. This captures all glyphs
                // including fallback-rendered ones without re-rasterizing.
                let (sr, sg, sb, sa) = *color;
                let grow = (*sigma * 3.0).ceil().max(0.0) as u32;
                let sw = tw_u32 + grow * 2;
                let sh = th_u32 + grow * 2;

                let mut shadow_rgba = RgbaImage::from_pixel(sw, sh, image::Rgba([sr, sg, sb, 0]));
                for (i, text_px) in text_pixmap.pixels().iter().enumerate() {
                    let ta = text_px.alpha() as u32;
                    if ta > 0 {
                        let final_a = ((sa as u32 * ta) / 255) as u8;
                        let x = (i as u32) % tw_u32 + grow;
                        let y = (i as u32) / tw_u32 + grow;
                        shadow_rgba.put_pixel(x, y, image::Rgba([sr, sg, sb, final_a]));
                    }
                }

                let shadow_dyn = DynamicImage::ImageRgba8(shadow_rgba);
                let blurred_shadow =
                    if *sigma > 0.0 { shadow_dyn.blur(*sigma) } else { shadow_dyn };
                let blurred_rgba = blurred_shadow.to_rgba8();

                let mut shadow_pixmap = new_pixmap(sw, sh, "shadow")?;
                for (src, dst) in
                    blurred_rgba.pixels().iter().zip(shadow_pixmap.pixels_mut().iter_mut())
                {
                    *dst = ColorU8::from_rgba(src[0], src[1], src[2], src[3]).premultiply();
                }

                let shadow_transform = Transform::from_translate(
                    start_x + dx - grow as f32,
                    start_y + dy - grow as f32,
                )
                .pre_concat(Transform::from_rotate_at(
                    config.rotation,
                    text_w / 2.0 + grow as f32,
                    text_h / 2.0 + grow as f32,
                ));
                let mut layer = new_pixmap(img_w, img_h, "shadow layer")?;
                layer.draw_pixmap(0, 0, shadow_pixmap.as_ref(), &paint, shadow_transform, None);
                Some(layer)
            }
            _ => None,
        }
    } else {
        None
    };

    // 3. Render / displace layers according to configuration
    match &config.displacement {
        DisplacementEffect::None => {
            if let Some(el) = effect_layer {
                main_pixmap.draw_pixmap(0, 0, el.as_ref(), &paint, Transform::identity(), None);
            }
            main_pixmap.draw_pixmap(0, 0, text_pixmap.as_ref(), &paint, transform, None);
        }
        DisplacementEffect::Explode { strength, binding } => {
            // Compute the centroid of all rendered glyphs in the text pixmap
            let pixels = text_pixmap.pixels();
            let mut sum_x: f64 = 0.0;
            let mut sum_y: f64 = 0.0;
            let mut count: u64 = 0;
            for py in 0..th_u32 {
                for px in 0..tw_u32 {
                    if pixels[(py * tw_u32 + px) as usize].alpha() > 0 {
                        sum_x += px as f64;
                        sum_y += py as f64;
                        count += 1;
                    }
                }
            }

            if count > 0 {
                let cx_local = (sum_x / count as f64) as f32;
                let cy_local = (sum_y / count as f64) as f32;

                let angle_rad = config.rotation.to_radians();
                let cos_a = angle_rad.cos();
                let sin_a = angle_rad.sin();
                let rel_x = cx_local - text_w / 2.0;
                let rel_y = cy_local - text_h / 2.0;
                let cx = start_x + text_w / 2.0 + rel_x * cos_a - rel_y * sin_a;
                let cy = start_y + text_h / 2.0 + rel_x * sin_a + rel_y * cos_a;

                let text_diag = (actual_w * actual_w + actual_h * actual_h).sqrt();
                let effect_radius = text_diag / 2.0 + *strength * 2.0;

                // 1. Warp the background canvas
                apply_explode(&mut main_pixmap, cx, cy, effect_radius, *strength, img_w, img_h);

                // 2. Warp and composite glyph effect layer using the binding factor
                if let Some(mut el) = effect_layer {
                    apply_explode(
                        &mut el,
                        cx,
                        cy,
                        effect_radius,
                        *strength * *binding,
                        img_w,
                        img_h,
                    );
                    main_pixmap.draw_pixmap(0, 0, el.as_ref(), &paint, Transform::identity(), None);
                }

                // 3. Warp and composite text layer at 100% strength
                let mut text_layer = new_pixmap(img_w, img_h, "text layer")?;
                text_layer.draw_pixmap(0, 0, text_pixmap.as_ref(), &paint, transform, None);
                apply_explode(&mut text_layer, cx, cy, effect_radius, *strength, img_w, img_h);
                main_pixmap.draw_pixmap(
                    0,
                    0,
                    text_layer.as_ref(),
                    &paint,
                    Transform::identity(),
                    None,
                );
            } else {
                if let Some(el) = effect_layer {
                    main_pixmap.draw_pixmap(0, 0, el.as_ref(), &paint, Transform::identity(), None);
                }
                main_pixmap.draw_pixmap(0, 0, text_pixmap.as_ref(), &paint, transform, None);
            }
        }
        DisplacementEffect::Meltdown { strength, direction, binding } => {
            // Build proximity field from rendered text glyphs
            let pixels = text_pixmap.pixels();
            let mut text_mask_src = new_pixmap(tw_u32, th_u32, "meltdown text mask")?;
            {
                let mask_pixels = text_mask_src.pixels_mut();
                for (i, px) in pixels.iter().enumerate() {
                    if px.alpha() > 0 {
                        mask_pixels[i] =
                            PremultipliedColorU8::from_rgba(255, 255, 255, 255).unwrap();
                    }
                }
            }

            let mut canvas_mask = new_pixmap(img_w, img_h, "meltdown canvas mask")?;
            canvas_mask.draw_pixmap(0, 0, text_mask_src.as_ref(), &paint, transform, None);

            let mut mask_img = GrayImage::new(img_w, img_h);
            for (i, px) in canvas_mask.pixels().iter().enumerate() {
                let x = (i as u32) % img_w;
                let y = (i as u32) / img_w;
                mask_img.put_pixel(x, y, image::Luma([px.alpha()]));
            }
            let blur_sigma = (*strength * 0.4).max(1.0);
            let blurred_mask = DynamicImage::ImageLuma8(mask_img).blur(blur_sigma).to_luma8();

            let field: Vec<f32> =
                blurred_mask.pixels().iter().map(|px| px[0] as f32 / 255.0).collect();

            let is_omni = *direction < 0.0;
            let dir_rad = direction.to_radians();
            let fixed_dx = dir_rad.cos();
            let fixed_dy = dir_rad.sin();

            // 1. Warp the background canvas
            apply_meltdown(
                &mut main_pixmap,
                &field,
                is_omni,
                fixed_dx,
                fixed_dy,
                *strength,
                img_w,
                img_h,
            );

            // 2. Warp and composite glyph effect layer using the binding factor
            if let Some(mut el) = effect_layer {
                apply_meltdown(
                    &mut el,
                    &field,
                    is_omni,
                    fixed_dx,
                    fixed_dy,
                    *strength * *binding,
                    img_w,
                    img_h,
                );
                main_pixmap.draw_pixmap(0, 0, el.as_ref(), &paint, Transform::identity(), None);
            }

            // 3. Warp and composite text layer at 100% strength
            let mut text_layer = new_pixmap(img_w, img_h, "text layer")?;
            text_layer.draw_pixmap(0, 0, text_pixmap.as_ref(), &paint, transform, None);
            apply_meltdown(
                &mut text_layer,
                &field,
                is_omni,
                fixed_dx,
                fixed_dy,
                *strength,
                img_w,
                img_h,
            );
            main_pixmap.draw_pixmap(0, 0, text_layer.as_ref(), &paint, Transform::identity(), None);
        }
    }

    finish_render(image, &main_pixmap, precision);
    Ok(())
}

/// Draw a single rasterized glyph onto the text pixmap with alpha blending
fn draw_glyph_pixels(
    text_pixmap: &mut Pixmap,
    tw: u32,
    th: u32,
    img: &cosmic_text::SwashImage,
    gx: i32,
    gy: i32,
    color: (u8, u8, u8, u8),
) {
    let origin_x = gx;
    let origin_y = gy;

    for r in 0..img.placement.height as i32 {
        for c in 0..img.placement.width as i32 {
            let px_x = origin_x + c;
            let px_y = origin_y + r;

            if px_x < 0 || px_y < 0 || px_x >= tw as i32 || px_y >= th as i32 {
                continue;
            }

            let pixel_idx = (r * img.placement.width as i32 + c) as usize;

            // Don't trust the buffer to match the stride we expect for the
            // content type; a short buffer should skip, not panic.
            let needed = match img.content {
                cosmic_text::SwashContent::Mask => pixel_idx + 1,
                cosmic_text::SwashContent::SubpixelMask => pixel_idx * 3 + 3,
                cosmic_text::SwashContent::Color => pixel_idx * 4 + 4,
            };
            if needed > img.data.len() {
                continue;
            }

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
                    (img.data[start], img.data[start + 1], img.data[start + 2], img.data[start + 3])
                }
            };

            let target_alpha = color.3 as u32;
            let final_alpha = (src_a as u32 * target_alpha) / 255;

            if final_alpha > 0 {
                let pr = (src_r as u32 * final_alpha) / 255;
                let pg = (src_g as u32 * final_alpha) / 255;
                let pb = (src_b as u32 * final_alpha) / 255;

                let idx = (px_y as u32 * tw + px_x as u32) as usize;
                let dst = text_pixmap.pixels()[idx];

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
