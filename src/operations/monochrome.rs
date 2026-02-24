use crate::{error::MagickError, image::Image};
use image::{DynamicImage, GrayImage, Luma};

// 1.00 = mathematically perfect (aggressive, noisy shadows)
// 0.35 = ImageMagick legacy (solid blacks, crushed shadows)
const SHADOW_DIFFUSION: f32 = 0.40;

// Controls black speckles in light areas.
// Set this very low (e.g., 0.05) to get clean, spotless whites.
const HIGHLIGHT_DIFFUSION: f32 = 0.00;

//   0.0: Pure Math (No Jitter)
//  20.0 to 60.0: Subtle Breakup
//  80.0 to 140.0: The "Film Grain" Sweet Spot
// 150.0 to 255.0: Heavy Noise
const THRESHOLD_JITTER: f32 = 140.0;

pub fn monochrome(image: &mut Image) -> Result<(), MagickError> {
    let mut grayscaled = image.pixels.to_luma8();
    apply_blue_noise_scatter(&mut grayscaled);
    image.pixels = DynamicImage::ImageLuma8(grayscaled);
    Ok(())
}

pub fn apply_blue_noise_scatter(image: &mut GrayImage) {
    let width = image.width();
    let height = image.height();

    // Pad by 2 to safely handle x-1 and x+1 boundary math
    let mut errors = vec![0.0f32; (width + 2) as usize * height as usize];

    for y in 0..height {
        for x in 0..width {
            let idx = (y * (width + 2) + x + 1) as usize;

            // Current pixel value + accumulated error
            let original_luma = image.get_pixel(x, y).0[0] as f32;
            let current_val = original_luma + errors[idx];

            // 1. Fetch blue noise (0 to 255) and normalize to (-0.5 to 0.5)
            let noise_u8 = get_noise(x, y);
            let noise = (noise_u8 as f32 / 255.0) - 0.5;

            // 2. Modulate the threshold with blue noise
            let dynamic_threshold = 127.5 + (noise * THRESHOLD_JITTER);

            // 3. Thresholding against the jittered baseline
            let (new_val, color) = if current_val > dynamic_threshold {
                (255.0, Luma([255]))
            } else {
                (0.0, Luma([0]))
            };

            image.put_pixel(x, y, color);

            // 4. Calculate raw error mathematically (guarantees brightness retention)
            let raw_error = current_val - new_val;

            // 5. Apply asymmetric damping for spotless whites and clean blacks
            let tuned_error = if raw_error > 0.0 {
                raw_error * SHADOW_DIFFUSION
            } else {
                raw_error * HIGHLIGHT_DIFFUSION
            };

            // 6. Tight Sierra Lite distribution (keeps grain microscopic)
            if x < width - 1 {
                errors[idx + 1] += tuned_error * 0.5; // Right: 1/2
            }
            if y < height - 1 {
                errors[idx + (width + 2) as usize - 1] += tuned_error * 0.25; // Bottom-Left: 1/4
                errors[idx + (width + 2) as usize] += tuned_error * 0.25; // Bottom: 1/4
            }
        }
    }
}

// Generate blue noise data with the following sequence of commands:
// * Clone or checkout the following commit:
//   https://github.com/mblode/blue-noise-rust/commit/aea756b5853828ac6401937ee39bea27b2f39898
// * Modify the src/generator.rs file to output raw bytes instead of PNG, e.g. by editing the
//   `save_blue_noise_to_png` to contain the following code:
//
//  ```
//  let raw = img.into_raw();
//  BufWriter::new(File::create(&filename).expect("do so"))
//    .write_all(&raw)
//    .expect("write failed");
//  ```
//  * Run `cargo build --release`
//  * Generate the blue noise file with:
//    `./target/release/blue-noise generate --size 64 --output blue-noise.bin`
//  * Copy the binary next to this source file.
const NOISE_DATA: &[u8] = include_bytes!("blue-noise-256.bin");
const NOISE_DATA_WIDTH_AND_HEIGHT: usize = 256;

/// Get the noise value at the given coordinates. If the coordinates are out of bounds,
/// they will wrap around. Means we don't need a noise texture as large as the image.
#[inline]
fn get_noise(x: u32, y: u32) -> u8 {
    let wrap_x = (x as usize) % NOISE_DATA_WIDTH_AND_HEIGHT;
    let wrap_y = (y as usize) % NOISE_DATA_WIDTH_AND_HEIGHT;
    NOISE_DATA[wrap_y * NOISE_DATA_WIDTH_AND_HEIGHT + wrap_x]
}
