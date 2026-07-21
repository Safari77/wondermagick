use crate::{error::MagickError, image::Image};
use image::DynamicImage;
use rayon::prelude::*;
use std::collections::HashSet;

/// Count the number of unique colors (RGB, ignoring alpha) in the image
/// and print the result to stdout.
/// Uses rayon parallel fold/reduce with HashSets for maximum throughput.
pub fn countcolors(image: &mut Image) -> Result<(), MagickError> {
    let unique_count = match &image.pixels {
        // 16-bit path: pack into u64 to prevent truncating data
        DynamicImage::ImageRgb16(_) | DynamicImage::ImageRgba16(_) => {
            let rgba = image.pixels.to_rgba16();
            let raw = rgba.as_raw();

            raw.par_chunks_exact(4)
                .fold(HashSet::<u64>::new, |mut set, px| {
                    let packed = (px[0] as u64) | ((px[1] as u64) << 16) | ((px[2] as u64) << 32);
                    set.insert(packed);
                    set
                })
                .reduce(HashSet::<u64>::new, |mut a, b| {
                    if a.len() >= b.len() {
                        a.extend(b);
                        a
                    } else {
                        let mut b = b;
                        b.extend(a);
                        b
                    }
                })
                .len()
        }
        // 8-bit path: pack into u32 for maximum hash/memory efficiency
        _ => {
            let rgba = image.pixels.to_rgba8();
            let raw = rgba.as_raw();

            raw.par_chunks_exact(4)
                .fold(HashSet::<u32>::new, |mut set, px| {
                    let packed = (px[0] as u32) | ((px[1] as u32) << 8) | ((px[2] as u32) << 16);
                    set.insert(packed);
                    set
                })
                .reduce(HashSet::<u32>::new, |mut a, b| {
                    if a.len() >= b.len() {
                        a.extend(b);
                        a
                    } else {
                        let mut b = b;
                        b.extend(a);
                        b
                    }
                })
                .len()
        }
    };

    println!("{}", unique_count);
    Ok(())
}
