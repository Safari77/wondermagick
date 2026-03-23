use crate::{arg_parsers::GrayscaleMethod, error::MagickError, image::Image, wm_err};
use image::{DynamicImage, GrayImage, Luma};

pub fn grayscale(image: &mut Image, method: &GrayscaleMethod) -> Result<(), MagickError> {
    match method {
        GrayscaleMethod::Rec709Luma => {
            // image-rs appears to be using something like Rec. 709 by default for grayscale
            // conversion of most image types.
            // https://github.com/image-rs/image/blob/2e121abff5f87028e85bf8f26a95f36f7b6182ac/src/images/buffer.rs#L1577
            // https://github.com/image-rs/image/issues/598
            image.pixels = image.pixels.grayscale();
            Ok(())
        }
        GrayscaleMethod::OklabLuminance => {
            let rgb = image.pixels.to_rgb8();
            let (width, height) = rgb.dimensions();
            let mut gray = GrayImage::new(width, height);

            for (x, y, pixel) in rgb.enumerate_pixels() {
                let oklab_l = srgb_to_oklab_l(pixel[0], pixel[1], pixel[2]);
                let luma = (oklab_l.clamp(0.0, 1.0) * 255.0).round() as u8;
                gray.put_pixel(x, y, Luma([luma]));
            }

            image.pixels = DynamicImage::ImageLuma8(gray);
            Ok(())
        }
        _ => Err(wm_err!(
            "grayscale method not implemented. Rec709Luma and OklabLuminance are the supported methods."
        )),
    }
}

/// Convert a single sRGB byte triplet to Oklab L (perceptual luminance in [0, 1]).
///
/// Pipeline: sRGB gamma → linear RGB → LMS (via Oklab matrix) → cube root → Oklab L.
/// Reference: Björn Ottosson, "A perceptual color space for image processing"
/// https://bottosson.github.io/posts/oklab/
#[inline]
fn srgb_to_oklab_l(r: u8, g: u8, b: u8) -> f64 {
    let r_lin = srgb_to_linear(r);
    let g_lin = srgb_to_linear(g);
    let b_lin = srgb_to_linear(b);

    // Linear sRGB → LMS (Oklab M1 matrix)
    let l = 0.4122214708 * r_lin + 0.5363325363 * g_lin + 0.0514459929 * b_lin;
    let m = 0.2119034982 * r_lin + 0.6806995451 * g_lin + 0.1073969566 * b_lin;
    let s = 0.0883024619 * r_lin + 0.2817188376 * g_lin + 0.6299787005 * b_lin;

    // Cube root nonlinearity
    let l_ = l.cbrt();
    let m_ = m.cbrt();
    let s_ = s.cbrt();

    // Oklab L (first row of M2 matrix)
    0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
}

/// sRGB gamma decoding: 8-bit sRGB → linear [0, 1].
#[inline]
fn srgb_to_linear(c: u8) -> f64 {
    let c = c as f64 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}
