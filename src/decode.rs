use std::io::{BufReader, Seek};

use image::{DynamicImage, ImageReaderOptions};

use crate::{
    arg_parsers::{FileFormat, Location},
    error::MagickError,
    image::{Image, InputProperties},
    wm_err, wm_try,
};

/// If the format has not been explicitly specified, guesses the format based on file contents.
pub fn decode(location: &Location, format: Option<FileFormat>) -> Result<Image, MagickError> {
    let format = match format {
        Some(FileFormat::IgnoreFile) => return Ok(blank_image(location)),
        Some(FileFormat::Format(fmt)) => Some(fmt),
        None => None,
    };

    let mut reader = match location {
        Location::Path(path) => ImageReaderOptions::open(path)
            .map_err(|error| wm_err!("unable to open image '{}': {error}", path.display()))?,
        Location::Stdio => {
            // The decoder requires Seek, which Stdout doesn't implement.
            // We copy stdin to a temporary file and open the file instead.
            let mut file = wm_try!(tempfile::tempfile());
            wm_try!(std::io::copy(&mut std::io::stdin(), &mut file));
            wm_try!(file.seek(std::io::SeekFrom::Start(0)));
            ImageReaderOptions::new(BufReader::new(file))
        }
    };

    let format = match format {
        Some(format) => {
            reader.set_format(format);
            Some(format)
        }
        None => {
            reader = wm_try!(reader.with_guessed_format());
            reader.format()
        }
    };
    // Disable the `image` crate's default decoding limits (notably the 512 MiB
    // `max_alloc` cap) so arbitrarily large images can be decoded.
    reader.no_limits();
    let mut reader = wm_try!(reader.into_reader());
    let (pixels, mut metadata) = wm_try!(reader.decode());
    let exif = metadata.exif_metadata().unwrap_or(None);
    let xmp = metadata.xmp_metadata().unwrap_or(None);
    let icc = metadata.icc_profile().unwrap_or(None);
    // `original_color_type` is now only an optional hint, set when the decoded data differs
    // from the source. When absent, the original matches the decoded image's color type.
    let color_type = metadata
        .attributes()
        .original_color_type
        .unwrap_or_else(|| pixels.color().into());
    let properties = InputProperties {
        filename: location.to_filename(),
        color_type,
    };
    Ok(Image {
        format,
        exif,
        xmp,
        icc,
        pixels,
        properties,
    })
}

pub fn blank_image(location: &Location) -> Image {
    Image {
        format: None,
        exif: None,
        icc: None,
        xmp: None,
        pixels: DynamicImage::new_rgb8(1, 1),
        properties: InputProperties {
            filename: location.to_filename(),
            color_type: image::ExtendedColorType::Rgb8,
        },
    }
}

// You know what would be a cool optimization for decoding process?
// Using the image thumbnail from the Exif metadata, when it's present.
// Many JPEG images have it; other formats can contain it as well.
// That way you don't even have to decode the full-size image!
// You can dump it from a JPEG that has one like this:
// $ exiftool -b -ThumbnailImage image.jpg > ~/thumb.jpg
// It saves so much time! Every competent thumbnailer uses it!
//
// But imagemagick is not a competent thumbnailer.
// Imagemagick not only ignores it, it does something far worse:
// It keeps the old thumbnail even after modifying the image.
// Making the embedded thumbnail WRONG.
//
// Not only does it not make use of the thumbnail that someone
// has already spent network bandwidth and storage on,
// but it also breaks any other program making use of it,
// by keeping the old thumbnail around
// even when it doesn't match the image anymore.
//
// This is completely wrong behavior no matter how you look at it.
// But that's what imagemagick does and what we must do as well.
// Until we decide enough is enough and some bugs aren't worth
// being compatible with, but that day is not today.
