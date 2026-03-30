use crate::{error::MagickError, image::Image};
use imageproc::contrast::{self, ThresholdType}; // <-- Add ThresholdType here

pub fn otsu(image: &mut Image) -> Result<(), MagickError> {
    let gray_img = image.pixels.to_luma8();
    let level = contrast::otsu_level(&gray_img);

    // Add ThresholdType::Binary as the 3rd argument
    let thresholded = contrast::threshold(&gray_img, level, ThresholdType::Binary);

    image.pixels = thresholded.into();
    Ok(())
}

pub fn kapur(image: &mut Image) -> Result<(), MagickError> {
    let gray_img = image.pixels.to_luma8();
    let level = contrast::kapur_level(&gray_img);

    // Add ThresholdType::Binary as the 3rd argument
    let thresholded = contrast::threshold(&gray_img, level, ThresholdType::Binary);

    image.pixels = thresholded.into();
    Ok(())
}

pub fn equalize_histogram(image: &mut Image) -> Result<(), MagickError> {
    let gray_img = image.pixels.to_luma8();
    let equalized = contrast::equalize_histogram(&gray_img);
    image.pixels = equalized.into();
    Ok(())
}
