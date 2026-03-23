use crate::{arg_parse_err::ArgParseErr, error::MagickError, image::Image};
use image::{DynamicImage, GrayImage};
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MorphologyMethod {
    Erode,
    Dilate,
    Open,
    Close,
    TopHat,
    BottomHat,
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
            "tophat" | "top-hat" | "top_hat" => MorphologyMethod::TopHat,
            "bottomhat" | "bottom-hat" | "bottom_hat" | "blackhat" | "black-hat" | "black_hat" => {
                MorphologyMethod::BottomHat
            }
            _ => {
                return Err(ArgParseErr::with_msg(
                    "invalid method. Use: erode, dilate, open, close, tophat, bottomhat",
                ))
            }
        };

        let shape = match parts[1].trim().to_lowercase().as_str() {
            "square" => KernelShape::Square,
            "cross" => KernelShape::Cross,
            "circle" | "disk" | "ellipse" => KernelShape::Circle,
            "rectangle" | "rect" => KernelShape::Rectangle,
            _ => {
                return Err(ArgParseErr::with_msg(
                    "invalid shape. Use: square, cross, circle, ellipse, rectangle",
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

        // Reject asymmetric sizes for shapes that require symmetry
        if kernel_width != kernel_height
            && shape != KernelShape::Rectangle
            && shape != KernelShape::Circle
        {
            return Err(ArgParseErr::with_msg(
                "asymmetric sizes (e.g., '3x7') are only allowed for 'rectangle' and 'circle' (ellipse) shapes",
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
        // Top-Hat: Original - Open(Image)
        // Extracts bright features smaller than the kernel from a dark background.
        MorphologyMethod::TopHat => {
            let eroded = apply_filter(&input, config, FilterType::Min);
            let opened = apply_filter(&eroded, config, FilterType::Max);
            subtract_images(&input, &opened)
        }
        // Bottom-Hat (Black-Hat): Close(Image) - Original
        // Extracts dark features (e.g. text) from a bright, unevenly lit background.
        MorphologyMethod::BottomHat => {
            let dilated = apply_filter(&input, config, FilterType::Max);
            let closed = apply_filter(&dilated, config, FilterType::Min);
            subtract_images(&closed, &input)
        }
    };

    image.pixels = DynamicImage::ImageLuma8(output);
    Ok(())
}

#[derive(Clone, Copy)]
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

    // Precompute ellipse semi-axis radii squared for the inclusion test.
    // For a symmetric kernel (half_w == half_h) this reduces to a circle.
    let rw_sq = (half_w * half_w).max(1) as f64;
    let rh_sq = (half_h * half_h).max(1) as f64;

    let shape = config.shape;
    let row_len = width as usize;
    let input_raw = input.as_raw();

    // Process rows in parallel. Each row writes to a disjoint slice of the output buffer
    // and reads only from the immutable input, so there are no data races.
    output
        .as_mut()
        .par_chunks_mut(row_len)
        .enumerate()
        .for_each(|(y_idx, row)| {
            let y = y_idx as u32;
            let y_start = y.saturating_sub(half_h);
            let y_end = (y + half_h).min(height - 1);

            for x in 0..width {
                let mut local_min = 255u8;
                let mut local_max = 0u8;

                let x_start = x.saturating_sub(half_w);
                let x_end = (x + half_w).min(width - 1);

                for ky in y_start..=y_end {
                    let row_offset = ky as usize * row_len;
                    for kx in x_start..=x_end {
                        let include = match shape {
                            KernelShape::Square | KernelShape::Rectangle => true,
                            KernelShape::Cross => kx == x || ky == y,
                            // Ellipse inclusion: (dx/rx)^2 + (dy/ry)^2 <= 1
                            // When half_w == half_h this is a circle.
                            KernelShape::Circle => {
                                let dx = kx.abs_diff(x) as f64;
                                let dy = ky.abs_diff(y) as f64;
                                (dx * dx) / rw_sq + (dy * dy) / rh_sq <= 1.0
                            }
                        };

                        if include {
                            let val = input_raw[row_offset + kx as usize];
                            if val < local_min {
                                local_min = val;
                            }
                            if val > local_max {
                                local_max = val;
                            }
                        }
                    }
                }

                row[x as usize] = match filter_type {
                    FilterType::Min => local_min,
                    FilterType::Max => local_max,
                };
            }
        });

    output
}

/// Saturating per-pixel subtraction: result = max(a - b, 0).
fn subtract_images(a: &GrayImage, b: &GrayImage) -> GrayImage {
    let (width, height) = a.dimensions();
    let mut output = GrayImage::new(width, height);

    let a_raw = a.as_raw();
    let b_raw = b.as_raw();

    output
        .as_mut()
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, out)| {
            *out = a_raw[i].saturating_sub(b_raw[i]);
        });

    output
}
