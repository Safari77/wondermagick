mod auto_orient;
mod blur;
mod combine;
mod crop;
mod flip;
pub use flip::Axis;
mod grayscale;
mod identify;
mod monochrome;
pub use monochrome::MonochromeConfig;
mod morphology;
mod sauvola;
pub use morphology::MorphologyConfig;
pub use sauvola::SauvolaConfig;
mod wolf_jolion;
pub use wolf_jolion::WolfJolionConfig;
mod phansalkar;
pub use phansalkar::PhansalkarConfig;
mod connected_components;
pub use connected_components::ConnectedComponentsConfig;
mod negate;
mod prune;
pub use prune::PruneConfig;
mod despeckle;
mod resize;
mod skeleton;
mod unsharpen;
pub use despeckle::DespeckleConfig;
mod normalize_background;
pub use normalize_background::NormalizeBackgroundConfig;
mod bm3d;
pub use bm3d::Bm3dConfig;
mod contrast;
mod quantize;
pub use quantize::QuantizeConfig;
mod text;
pub use text::TextConfig;

use crate::{
    arg_parsers::{
        BlurGeometry, CropGeometry, Filter, GrayscaleMethod, IdentifyFormat, LoadCropGeometry,
        ResizeGeometry, UnsharpenGeometry,
    },
    error::MagickError,
    image::Image,
    plan,
};

#[derive(Debug, Clone, PartialEq)]
pub enum Operation {
    Resize(ResizeGeometry, Option<Filter>),
    Thumbnail(ResizeGeometry, Option<Filter>),
    Scale(ResizeGeometry),
    Sample(ResizeGeometry),
    CropOnLoad(LoadCropGeometry),
    Crop(CropGeometry),
    Despeckle(DespeckleConfig),
    Identify(Option<IdentifyFormat>),
    Negate,
    NormalizeBackground(NormalizeBackgroundConfig),
    AutoOrient,
    Blur(BlurGeometry),
    GaussianBlur(BlurGeometry),
    Grayscale(GrayscaleMethod),
    Flip(Axis),
    Monochrome(MonochromeConfig),
    Unsharpen(UnsharpenGeometry),
    Sauvola(SauvolaConfig),
    WolfJolion(WolfJolionConfig),
    Phansalkar(PhansalkarConfig),
    ConnectedComponents(ConnectedComponentsConfig),
    Morphology(MorphologyConfig),
    Skeleton,
    Prune(PruneConfig),
    Bm3d(Bm3dConfig),
    Otsu,
    Kapur,
    EqualizeHistogram,
    Quantize(QuantizeConfig),
    Text(TextConfig),
}

impl Operation {
    pub fn execute(&self, image: &mut Image) -> Result<(), MagickError> {
        match self {
            Operation::Resize(geom, filter) => resize::resize(image, geom, *filter),
            Operation::Thumbnail(geom, filter) => resize::thumbnail(image, geom, *filter),
            Operation::Scale(geom) => resize::scale(image, geom),
            Operation::Sample(geom) => resize::sample(image, geom),
            Operation::CropOnLoad(geom) => crop::crop_on_load(image, geom),
            Operation::Crop(geom) => crop::crop(image, geom),
            Operation::Despeckle(config) => despeckle::despeckle(image, config),
            Operation::Identify(format) => identify::identify(image, format.clone()),
            Operation::Negate => negate::negate(image),
            Operation::NormalizeBackground(config) => {
                normalize_background::normalize_background(image, config)
            }
            Operation::AutoOrient => auto_orient::auto_orient(image),
            Operation::Blur(geom) => blur::blur(image, geom),
            Operation::GaussianBlur(geom) => blur::gaussian_blur(image, geom),
            Operation::Grayscale(method) => grayscale::grayscale(image, method),
            Operation::Flip(axis) => flip::flip(image, axis),
            Operation::Monochrome(config) => monochrome::monochrome(image, config),
            Operation::Unsharpen(geom) => unsharpen::unsharpen(image, geom),
            Operation::Sauvola(config) => sauvola::sauvola(image, config),
            Operation::WolfJolion(config) => wolf_jolion::wolf_jolion(image, config),
            Operation::Phansalkar(config) => phansalkar::phansalkar(image, config),
            Operation::ConnectedComponents(config) => {
                connected_components::connected_components(image, config)
            }
            Operation::Morphology(config) => morphology::morphology(image, config),
            Operation::Skeleton => skeleton::skeleton(image),
            Operation::Prune(config) => prune::prune(image, config),
            Operation::Bm3d(config) => bm3d::bm3d(image, config),
            Operation::Otsu => contrast::otsu(image),
            Operation::Kapur => contrast::kapur(image),
            Operation::EqualizeHistogram => contrast::equalize_histogram(image),
            Operation::Quantize(config) => quantize::quantize(image, config),
            Operation::Text(config) => text::render_text(image, config),
        }
    }

    /// Modifiers are flags such as -quality that affect operations.
    /// For global operations we need to alter them after the operation's creation,
    /// to apply up-to-date modifiers.
    pub fn apply_modifiers(&mut self, mods: &plan::Modifiers) {
        use Operation::*;
        match self {
            Resize(resize_geometry, _) => *self = Resize(*resize_geometry, mods.filter),
            Thumbnail(resize_geometry, _) => *self = Thumbnail(*resize_geometry, mods.filter),
            Scale(_) => (),
            Sample(_) => (),
            CropOnLoad(_) => (),
            Crop(_) => (),
            Despeckle(_) => (),
            Identify(_) => *self = Identify(mods.identify_format.clone()),
            Negate => (),
            NormalizeBackground(_) => (),
            AutoOrient => (),
            Blur(_) => (),
            GaussianBlur(_) => (),
            Grayscale(_) => (),
            Flip(_) => (),
            Monochrome(_) => (),
            Unsharpen(_) => (),
            Sauvola(_) => (),
            WolfJolion(_) => (),
            Phansalkar(_) => (),
            ConnectedComponents(_) => (),
            Morphology(_) => (),
            Skeleton => (),
            Prune(_) => (),
            Bm3d(_) => (),
            Otsu => (),
            Kapur => (),
            EqualizeHistogram => (),
            Quantize(_) => (),
            Text(_) => (),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum RewriteOperation {
    Combine {
        color: image::ColorType,
        /// Rewrite the color model to true color (`sRGB`) when the channel count is exceeded?
        fallback_for_channel_count: bool,
    },
}

impl RewriteOperation {
    pub(crate) fn execute(&self, sequence: &mut Vec<Image>) -> Result<(), MagickError> {
        match self {
            &RewriteOperation::Combine {
                color,
                fallback_for_channel_count,
            } => {
                let image =
                    combine::combine(sequence.split_off(0), color, fallback_for_channel_count)?;
                sequence.push(image);
                Ok(())
            }
        }
    }
}
