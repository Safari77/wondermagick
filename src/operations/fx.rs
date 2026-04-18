use crate::{arg_parse_err::ArgParseErr, error::MagickError, image::Image};
use image::DynamicImage;

#[derive(Debug, Clone, PartialEq)]
pub enum FxOperator {
    LessThanEqual,
    GreaterThanEqual,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FxMode {
    Rgb {
        cond_r: u8,
        cond_g: u8,
        cond_b: u8,
        replace_r: u8,
        replace_g: u8,
        replace_b: u8,
    },
    Gray {
        cond: u8,
        replace: u8,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct FxConfig {
    pub op: FxOperator,
    pub mode: FxMode,
}

impl FxConfig {
    /// Parse from a comma-separated string.
    /// Grayscale format (3 args): "operator,cond,replace" (e.g., "le,10,0")
    /// RGB format (7 args): "operator,cond_r,cond_g,cond_b,replace_r,replace_g,replace_b" (e.g., "le,5,5,5,0,0,0")
    pub fn parse_arg(s: &str) -> Result<Self, ArgParseErr> {
        let s = s.trim();
        let parts: Vec<&str> = s.split(',').collect();

        let len = parts.len();
        if len != 3 && len != 7 {
            return Err(ArgParseErr::with_msg(
                "fx requires either 3 values for Grayscale (op,val,replace) or 7 values for RGB (op,r,g,b,repl_r,repl_g,repl_b)",
            ));
        }

        let op_str = parts[0].trim().to_lowercase();
        let op = match op_str.as_str() {
            "le" | "<=" => FxOperator::LessThanEqual,
            "ge" | ">=" => FxOperator::GreaterThanEqual,
            _ => {
                return Err(ArgParseErr::with_msg(
                    "fx operator must be 'le' (<=) or 'ge' (>=)",
                ))
            }
        };

        let mode = if len == 7 {
            let cond_r = parts[1]
                .trim()
                .parse::<u8>()
                .map_err(|_| ArgParseErr::with_msg("invalid condition r value"))?;
            let cond_g = parts[2]
                .trim()
                .parse::<u8>()
                .map_err(|_| ArgParseErr::with_msg("invalid condition g value"))?;
            let cond_b = parts[3]
                .trim()
                .parse::<u8>()
                .map_err(|_| ArgParseErr::with_msg("invalid condition b value"))?;
            let replace_r = parts[4]
                .trim()
                .parse::<u8>()
                .map_err(|_| ArgParseErr::with_msg("invalid replace r value"))?;
            let replace_g = parts[5]
                .trim()
                .parse::<u8>()
                .map_err(|_| ArgParseErr::with_msg("invalid replace g value"))?;
            let replace_b = parts[6]
                .trim()
                .parse::<u8>()
                .map_err(|_| ArgParseErr::with_msg("invalid replace b value"))?;

            FxMode::Rgb {
                cond_r,
                cond_g,
                cond_b,
                replace_r,
                replace_g,
                replace_b,
            }
        } else {
            let cond = parts[1]
                .trim()
                .parse::<u8>()
                .map_err(|_| ArgParseErr::with_msg("invalid grayscale condition value"))?;
            let replace = parts[2]
                .trim()
                .parse::<u8>()
                .map_err(|_| ArgParseErr::with_msg("invalid grayscale replace value"))?;

            FxMode::Gray { cond, replace }
        };

        Ok(Self { op, mode })
    }
}

pub fn fx(image: &mut Image, config: &FxConfig) -> Result<(), MagickError> {
    let is_color = image.pixels.color().has_color();

    match &config.mode {
        FxMode::Rgb {
            cond_r,
            cond_g,
            cond_b,
            replace_r,
            replace_g,
            replace_b,
        } => {
            if !is_color {
                return Err(crate::wm_err!("fx RGB mode (7 arguments) requires an RGB image, but a grayscale image was loaded."));
            }

            let mut output = image.pixels.to_rgba8();
            for pixel in output.pixels_mut() {
                let condition_met = match config.op {
                    FxOperator::LessThanEqual => {
                        pixel[0] <= *cond_r && pixel[1] <= *cond_g && pixel[2] <= *cond_b
                    }
                    FxOperator::GreaterThanEqual => {
                        pixel[0] >= *cond_r && pixel[1] >= *cond_g && pixel[2] >= *cond_b
                    }
                };

                if condition_met {
                    pixel[0] = *replace_r;
                    pixel[1] = *replace_g;
                    pixel[2] = *replace_b;
                }
            }
            image.pixels = DynamicImage::ImageRgba8(output);
        }
        FxMode::Gray { cond, replace } => {
            if is_color {
                return Err(crate::wm_err!("fx Grayscale mode (3 arguments) requires a grayscale image, but an RGB image was loaded."));
            }

            let mut output = image.pixels.to_luma8();
            for pixel in output.pixels_mut() {
                let condition_met = match config.op {
                    FxOperator::LessThanEqual => pixel[0] <= *cond,
                    FxOperator::GreaterThanEqual => pixel[0] >= *cond,
                };

                if condition_met {
                    pixel[0] = *replace;
                }
            }
            image.pixels = DynamicImage::ImageLuma8(output);
        }
    }

    Ok(())
}
