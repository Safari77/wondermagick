use crate::{arg_parse_err::ArgParseErr, error::MagickError, image::Image};
use image::{DynamicImage, GrayImage, Luma};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MorphologyMethod {
    Erode,
    Dilate,
    Open,
    Close,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KernelShape {
    Square,
    Cross,
    Circle,
    Rectangle,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MorphologyConfig {
    pub method: MorphologyMethod,
    pub shape: KernelShape,
    pub kernel_width: u32,
    pub kernel_height: u32,
}

impl Default for MorphologyConfig {
    fn default() -> Self {
        Self {
            method: MorphologyMethod::Close,
            shape: KernelShape::Square,
            kernel_width: 3,
            kernel_height: 3,
        }
    }
}

impl MorphologyConfig {
    pub fn parse_arg(s: &str) -> Result<Self, ArgParseErr> {
        let s = s.trim();
        if s.eq_ignore_ascii_case("default") {
            return Ok(Self::default());
        }

        let parts: Vec<&str> = s.split(',').collect();
        if parts.len() != 3 {
            return Err(ArgParseErr::with_msg(
                "morphology requires exactly 3 comma-separated values: method,shape,size (e.g., 'close,rectangle,1x5')",
            ));
        }

        let method = match parts[0].trim().to_lowercase().as_str() {
            "erode" => MorphologyMethod::Erode,
            "dilate" => MorphologyMethod::Dilate,
            "open" => MorphologyMethod::Open,
            "close" => MorphologyMethod::Close,
            _ => {
                return Err(ArgParseErr::with_msg(
                    "invalid method. Use: erode, dilate, open, close",
                ))
            }
        };

        let shape = match parts[1].trim().to_lowercase().as_str() {
            "square" => KernelShape::Square,
            "cross" => KernelShape::Cross,
            "circle" | "disk" => KernelShape::Circle,
            "rectangle" | "rect" => KernelShape::Rectangle,
            _ => {
                return Err(ArgParseErr::with_msg(
                    "invalid shape. Use: square, cross, circle, rectangle",
                ))
            }
        };

        // Parse size, supporting both "3" (3x3) and "1x5" (width x height)
        let size_str = parts[2].trim().to_lowercase();
        let (kernel_width, kernel_height) = if size_str.contains('x') {
            let dims: Vec<&str> = size_str.split('x').collect();
            if dims.len() != 2 {
                return Err(ArgParseErr::with_msg("invalid WxH size format"));
            }
            let w = dims[0]
                .parse::<u32>()
                .map_err(|_| ArgParseErr::with_msg("invalid width"))?;
            let h = dims[1]
                .parse::<u32>()
                .map_err(|_| ArgParseErr::with_msg("invalid height"))?;
            (w, h)
        } else {
            let s = size_str
                .parse::<u32>()
                .map_err(|_| ArgParseErr::with_msg("invalid size"))?;
            (s, s)
        };

        // Reject asymmetric sizes for symmetric shapes
        if kernel_width != kernel_height && shape != KernelShape::Rectangle {
            return Err(ArgParseErr::with_msg(
                "asymmetric sizes (e.g., '1x10') are only allowed when using the 'rectangle' shape",
            ));
        }

        // Ensure both dimensions are odd numbers
        if kernel_width % 2 == 0 || kernel_height % 2 == 0 {
            return Err(ArgParseErr::with_msg(
                "kernel dimensions must be odd numbers (e.g., 3, 5, or 1x5) to have a true center pixel",
            ));
        }

        // Protect against massive kernels
        if kernel_width > 255 || kernel_height > 255 {
            return Err(ArgParseErr::with_msg(
                "kernel size is too large (maximum allowed dimension is 255)",
            ));
        }

        Ok(Self {
            method,
            shape,
            kernel_width,
            kernel_height,
        })
    }
}

pub fn morphology(image: &mut Image, config: &MorphologyConfig) -> Result<(), MagickError> {
    let input = image.pixels.to_luma8();

    let output = match config.method {
        MorphologyMethod::Erode => apply_filter(&input, config, FilterType::Min),
        MorphologyMethod::Dilate => apply_filter(&input, config, FilterType::Max),
        MorphologyMethod::Open => {
            let eroded = apply_filter(&input, config, FilterType::Min);
            apply_filter(&eroded, config, FilterType::Max)
        }
        MorphologyMethod::Close => {
            let dilated = apply_filter(&input, config, FilterType::Max);
            apply_filter(&dilated, config, FilterType::Min)
        }
    };

    image.pixels = DynamicImage::ImageLuma8(output);
    Ok(())
}

enum FilterType {
    Min,
    Max,
}

fn apply_filter(
    input: &GrayImage,
    config: &MorphologyConfig,
    filter_type: FilterType,
) -> GrayImage {
    let (width, height) = input.dimensions();
    let mut output = GrayImage::new(width, height);

    let half_w = config.kernel_width / 2;
    let half_h = config.kernel_height / 2;

    // Fallback circle radius calculation
    let r_sq = std::cmp::max(half_w, half_h) * std::cmp::max(half_w, half_h);

    for y in 0..height {
        for x in 0..width {
            let mut local_min = 255u8;
            let mut local_max = 0u8;

            let x_start = x.saturating_sub(half_w);
            let x_end = (x + half_w).min(width - 1);
            let y_start = y.saturating_sub(half_h);
            let y_end = (y + half_h).min(height - 1);

            for ky in y_start..=y_end {
                for kx in x_start..=x_end {
                    let include = match config.shape {
                        KernelShape::Square | KernelShape::Rectangle => true,
                        KernelShape::Cross => kx == x || ky == y,
                        KernelShape::Circle => {
                            let dx = kx.abs_diff(x);
                            let dy = ky.abs_diff(y);
                            (dx * dx + dy * dy) <= r_sq
                        }
                    };

                    if include {
                        let val = input.get_pixel(kx, ky)[0];
                        if val < local_min {
                            local_min = val;
                        }
                        if val > local_max {
                            local_max = val;
                        }
                    }
                }
            }

            let result_val = match filter_type {
                FilterType::Min => local_min,
                FilterType::Max => local_max,
            };

            output.put_pixel(x, y, Luma([result_val]));
        }
    }

    output
}
