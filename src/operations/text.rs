use crate::{arg_parse_err::ArgParseErr, error::MagickError, image::Image};
use cosmic_text::{
    Align, Attrs, Buffer, Family, FontSystem, Metrics, Shaping, SwashCache,
};
use image::{DynamicImage, RgbaImage};
use tiny_skia::{BlendMode, ColorU8, Pixmap, PixmapPaint, PremultipliedColorU8, Transform};

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
pub struct TextConfig {
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
    /// Format: "text,font_name,font_size,color,rotation,justify,x,y"
    /// Example: "Hello\nWorld,Arial,5%,#FF000080,-45.0,center,center,80%"
    /// Position units: px (absolute), % (0%=start, 100%=end-aligned), em (font-size-relative), center
    pub fn parse_arg(s: &str) -> Result<Self, ArgParseErr> {
        let mut parts: Vec<&str> = s.rsplitn(8, ',').collect();
        if parts.len() != 8 {
            return Err(ArgParseErr::with_msg(
                "text requires exactly 8 comma-separated values: \
                 text,font_name,font_size,color,rotation,justify,x,y",
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
}

pub fn render_text(image: &mut Image, config: &TextConfig) -> Result<(), MagickError> {
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

    main_pixmap.draw_pixmap(
        0,
        0,
        text_pixmap.as_ref(),
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
