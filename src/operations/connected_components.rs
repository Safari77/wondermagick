use crate::{arg_parse_err::ArgParseErr, error::MagickError, image::Image};
use image::{DynamicImage, GrayImage, Luma};

#[derive(Debug, Clone, PartialEq)]
pub struct ConnectedComponentsConfig {
    pub area_threshold: u32,
}

impl Default for ConnectedComponentsConfig {
    fn default() -> Self {
        Self { area_threshold: 10 }
    }
}

impl ConnectedComponentsConfig {
    /// Parse from a single integer string, or "default".
    /// Example: "10" or "default"
    pub fn parse_arg(s: &str) -> Result<Self, ArgParseErr> {
        let s = s.trim();
        if s.eq_ignore_ascii_case("default") {
            return Ok(Self::default());
        }

        let area_threshold = s.parse::<u32>().map_err(|_| {
            ArgParseErr::with_msg("invalid area_threshold (must be a positive integer)")
        })?;

        Ok(Self { area_threshold })
    }
}

pub fn connected_components(
    image: &mut Image,
    config: &ConnectedComponentsConfig,
) -> Result<(), MagickError> {
    let input = image.pixels.to_luma8();
    let (width, height) = input.dimensions();

    // Initialize the output image entirely with white (background)
    let mut output = GrayImage::from_pixel(width, height, Luma([255]));

    // Keep track of visited pixels to avoid infinite loops
    let mut visited = vec![false; (width * height) as usize];

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;

            // If it's already visited or it's a white pixel (background), skip it
            if visited[idx] || input.get_pixel(x, y)[0] != 0 {
                continue;
            }

            // We found a new, unvisited black pixel. Start a DFS to find the whole blob.
            let mut blob = Vec::new();
            let mut stack = vec![(x, y)];
            visited[idx] = true;

            while let Some((cx, cy)) = stack.pop() {
                blob.push((cx, cy));

                // 4-way connectivity (Up, Down, Left, Right)
                // Wrapping sub handles the 0 boundary safely (results in u32::MAX)
                let neighbors = [
                    (cx.wrapping_sub(1), cy),
                    (cx + 1, cy),
                    (cx, cy.wrapping_sub(1)),
                    (cx, cy + 1),
                ];

                for &(nx, ny) in &neighbors {
                    // Because wrapping_sub underflows to u32::MAX, the < width check safely
                    // handles both the top/left bounds (0) and bottom/right bounds.
                    if nx < width && ny < height {
                        let n_idx = (ny * width + nx) as usize;

                        // If the neighbor is black and unvisited, add it to the stack
                        if !visited[n_idx] && input.get_pixel(nx, ny)[0] == 0 {
                            visited[n_idx] = true;
                            stack.push((nx, ny));
                        }
                    }
                }
            }

            // DFS is done. If the blob is large enough, paint it black on the output image.
            // If it is too small, we do nothing (leaving it as the default white background).
            if blob.len() as u32 >= config.area_threshold {
                for (bx, by) in blob {
                    output.put_pixel(bx, by, Luma([0]));
                }
            }
        }
    }

    image.pixels = DynamicImage::ImageLuma8(output);
    Ok(())
}
