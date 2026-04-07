use crate::{error::MagickError, image::Image};
use rayon::prelude::*;
use std::collections::HashSet;

/// Count the number of unique colors (RGB, ignoring alpha) in the image
/// and print the result to stdout.
/// Uses rayon parallel fold/reduce with HashSets for maximum throughput.
pub fn countcolors(image: &mut Image) -> Result<(), MagickError> {
    let rgba = image.pixels.to_rgba8();
    let raw = rgba.as_raw();

    // Pack each pixel's RGB into a u32 (ignore alpha to match gmic behavior)
    // Process in parallel: each thread builds a local HashSet, then merge
    let unique_count = raw
        .par_chunks_exact(4)
        .fold(
            HashSet::<u32>::new,
            |mut set, px| {
                let packed = (px[0] as u32) | ((px[1] as u32) << 8) | ((px[2] as u32) << 16);
                set.insert(packed);
                set
            },
        )
        .reduce(
            HashSet::<u32>::new,
            |mut a, b| {
                // always extend the larger set to minimize re-hashing
                if a.len() >= b.len() {
                    a.extend(b);
                    a
                } else {
                    let mut b = b;
                    b.extend(a);
                    b
                }
            },
        )
        .len();

    println!("{}", unique_count);
    Ok(())
}
