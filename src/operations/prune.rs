use crate::{arg_parse_err::ArgParseErr, error::MagickError, image::Image};
use image::{DynamicImage, GrayImage, Luma};

#[derive(Debug, Clone, PartialEq)]
pub struct PruneConfig {
    pub iterations: u32,
}

impl Default for PruneConfig {
    fn default() -> Self {
        Self { iterations: 3 }
    }
}

impl PruneConfig {
    /// Parse from a single integer string, or "default".
    /// Example: "5" or "default"
    pub fn parse_arg(s: &str) -> Result<Self, ArgParseErr> {
        let s = s.trim();
        if s.eq_ignore_ascii_case("default") {
            return Ok(Self::default());
        }

        let iterations = s.parse::<u32>().map_err(|_| {
            ArgParseErr::with_msg("invalid prune iterations (must be a positive integer)")
        })?;

        Ok(Self { iterations })
    }
}

/// Iteratively removes endpoints (pixels with exactly 1 neighbor) from a skeletonized image.
pub fn prune(image: &mut Image, config: &PruneConfig) -> Result<(), MagickError> {
    let input = image.pixels.to_luma8();
    let (width, height) = input.dimensions();

    // Map to a binary grid (1 = foreground/black, 0 = background/white)
    let mut grid = vec![0u8; (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            if input.get_pixel(x, y)[0] < 128 {
                grid[(y * width + x) as usize] = 1;
            }
        }
    }

    let mut endpoints = Vec::new();

    let get_pixel = |g: &[u8], x: i32, y: i32| -> u8 {
        if x < 0 || x >= width as i32 || y < 0 || y >= height as i32 {
            0
        } else {
            g[(y * width as i32 + x) as usize]
        }
    };

    for _ in 0..config.iterations {
        endpoints.clear();

        for y in 0..height as i32 {
            for x in 0..width as i32 {
                if get_pixel(&grid, x, y) == 1 {
                    // Count the 8-connected neighbors
                    let neighbors = get_pixel(&grid, x - 1, y - 1)
                        + get_pixel(&grid, x, y - 1)
                        + get_pixel(&grid, x + 1, y - 1)
                        + get_pixel(&grid, x - 1, y)
                        + get_pixel(&grid, x + 1, y)
                        + get_pixel(&grid, x - 1, y + 1)
                        + get_pixel(&grid, x, y + 1)
                        + get_pixel(&grid, x + 1, y + 1);

                    // If a pixel has exactly 1 neighbor, it's the tip of a branch.
                    if neighbors == 1 {
                        endpoints.push((x, y));
                    }
                }
            }
        }

        // If no endpoints were found, the skeleton is completely closed (a loop) or empty.
        if endpoints.is_empty() {
            break;
        }

        // Erase the endpoints
        for &(x, y) in &endpoints {
            grid[(y * width as i32 + x) as usize] = 0;
        }
    }

    // Convert back to Wondermagick GrayImage
    let mut output = GrayImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let val = if grid[(y * width + x) as usize] == 1 {
                0
            } else {
                255
            };
            output.put_pixel(x, y, Luma([val]));
        }
    }

    image.pixels = DynamicImage::ImageLuma8(output);
    Ok(())
}
