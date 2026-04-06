use crate::{error::MagickError, image::Image};
use image::{DynamicImage, GrayImage, Luma};

/// Applies the Zhang-Suen skeletonization (thinning) algorithm to a binary image.
pub fn skeleton(image: &mut Image) -> Result<(), MagickError> {
    let input = image.pixels.to_luma8();
    let (width, height) = input.dimensions();

    // Convert the image to a mutable boolean/binary grid.
    // Zhang-Suen assumes Foreground = 1, Background = 0.
    // In Wondermagick, Black (0) is usually foreground, White (255) is background.
    // We treat anything darker than 128 as foreground.
    let mut grid = vec![0u8; (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            if input.get_pixel(x, y)[0] < 128 {
                grid[(y * width + x) as usize] = 1;
            }
        }
    }

    let mut has_changed = true;
    let mut markers = Vec::new();

    // Helper closure to safely get pixel value (out-of-bounds is treated as background '0')
    let get_pixel = |g: &[u8], x: i32, y: i32| -> u8 {
        if x < 0 || x >= width as i32 || y < 0 || y >= height as i32 {
            0
        } else {
            g[(y * width as i32 + x) as usize]
        }
    };

    while has_changed {
        has_changed = false;

        // --- PASS 1 ---
        markers.clear();
        for y in 0..height as i32 {
            for x in 0..width as i32 {
                if get_pixel(&grid, x, y) == 1 && zhang_suen_step(&grid, x, y, get_pixel, 1) {
                    markers.push((x, y));
                }
            }
        }
        if !markers.is_empty() {
            has_changed = true;
            for &(x, y) in &markers {
                grid[(y * width as i32 + x) as usize] = 0;
            }
        }

        // --- PASS 2 ---
        markers.clear();
        for y in 0..height as i32 {
            for x in 0..width as i32 {
                if get_pixel(&grid, x, y) == 1 && zhang_suen_step(&grid, x, y, get_pixel, 2) {
                    markers.push((x, y));
                }
            }
        }
        if !markers.is_empty() {
            has_changed = true;
            for &(x, y) in &markers {
                grid[(y * width as i32 + x) as usize] = 0;
            }
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

/// Evaluates the conditions for a single pixel during a Zhang-Suen pass.
fn zhang_suen_step<F>(grid: &[u8], x: i32, y: i32, get: F, pass: u8) -> bool
where
    F: Fn(&[u8], i32, i32) -> u8,
{
    // Clockwise neighbors starting from Top (P2)
    let p2 = get(grid, x, y - 1);
    let p3 = get(grid, x + 1, y - 1);
    let p4 = get(grid, x + 1, y);
    let p5 = get(grid, x + 1, y + 1);
    let p6 = get(grid, x, y + 1);
    let p7 = get(grid, x - 1, y + 1);
    let p8 = get(grid, x - 1, y);
    let p9 = get(grid, x - 1, y - 1);

    // Number of non-zero neighbors
    let b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;

    // Number of 0 -> 1 transitions in the sequence
    let mut a = 0;
    if p2 == 0 && p3 == 1 {
        a += 1;
    }
    if p3 == 0 && p4 == 1 {
        a += 1;
    }
    if p4 == 0 && p5 == 1 {
        a += 1;
    }
    if p5 == 0 && p6 == 1 {
        a += 1;
    }
    if p6 == 0 && p7 == 1 {
        a += 1;
    }
    if p7 == 0 && p8 == 1 {
        a += 1;
    }
    if p8 == 0 && p9 == 1 {
        a += 1;
    }
    if p9 == 0 && p2 == 1 {
        a += 1;
    }

    // Condition 1: 2 <= B(P1) <= 6
    if !(2..=6).contains(&b) {
        return false;
    }

    // Condition 2: A(P1) == 1
    if a != 1 {
        return false;
    }

    // Conditions 3 & 4 depend on the pass
    if pass == 1 {
        if (p2 * p4 * p6) != 0 || (p4 * p6 * p8) != 0 {
            return false;
        }
    } else {
        if (p2 * p4 * p8) != 0 || (p2 * p6 * p8) != 0 {
            return false;
        }
    }

    true
}
