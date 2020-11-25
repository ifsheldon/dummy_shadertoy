use std::fs::File;
use std::io;
use std::io::{BufReader, Error};

use image::{DynamicImage, GenericImageView, ImageError};
use image::io::{Reader as ImageReader, Reader};

use crate::data::Vec3;
use crate::err::ImageCreationError;
use crate::utils::to_color;

#[derive(PartialEq, Debug)]
pub enum Interpolation
{
    Bilinear,
    Nearest,
}

pub enum Tiling
{
    Repeat,
    Clamp,
}

pub struct Tex2D
{
    img: DynamicImage,
    width: u32,
    width_f: f32,
    height: u32,
    height_f: f32,
    interpolation: Interpolation,
    tiling: Tiling,
}

type ColorF = Vec3;
type ColorU8 = pixel_canvas::Color;

impl Tex2D
{
    pub fn from_file(path: String, interpolation: Interpolation, tiling: Tiling) -> Result<Self, ImageCreationError>
    {
        let img;
        match ImageReader::open(path) {
            Ok(reader) => {
                match reader.decode() {
                    Ok(image) => {
                        img = image;
                    }
                    Err(image_error) => {
                        return Err(ImageCreationError::ImageError(image_error));
                    }
                }
            }
            Err(io_error) => {
                return Err(ImageCreationError::IOError(io_error));
            }
        }
        Ok(Tex2D {
            width: img.width(),
            width_f: img.width() as f32,
            height: img.height(),
            height_f: img.height() as f32,
            img,
            interpolation,
            tiling,
        })
    }

    pub fn get_color_f(&self, u: f32, v: f32) -> ColorF {
        let (u, v) = self.check_uv(u, v);
        let x = u * (self.width_f - 1.0);
        let y = v * (self.height_f - 1.0);
        match self.interpolation {
            Interpolation::Bilinear => {
                let x_lower = x.floor();
                let x_upper = x_lower + 1.0;
                let x_frac = x.fract();
                let y_lower = y.floor();
                let y_upper = y_lower + 1.0;
                let y_frac = y.fract();
            }
            Interpolation::Nearest => {
                let x_round = x.round();
                let y_round = y.round();
            }
        }
        unimplemented!()
    }

    pub fn get_color_u8(&self, u: f32, v: f32) -> ColorU8 {
        to_color(self.get_color_f(u, v))
    }

    fn check_uv(&self, u: f32, v: f32) -> (f32, f32)
    {
        let u_in_bound = Self::in_bound(u);
        let v_in_bound = Self::in_bound(v);
        return if u_in_bound && v_in_bound {
            (u, v)
        } else if u_in_bound && !v_in_bound {
            (u, self.process_coord(v))
        } else if !u_in_bound && v_in_bound {
            (self.process_coord(u), v)
        } else {
            (self.process_coord(u), self.process_coord(v))
        };
    }

    #[inline]
    fn process_coord(&self, v: f32) -> f32 {
        return match self.tiling {
            Tiling::Repeat => v.fract(),
            Tiling::Clamp => if v < 0.0 { 0.0 } else { 1.0 },
        };
    }

    #[inline]
    fn in_bound(v: f32) -> bool
    {
        v >= 0.0 && v <= 1.0
    }
}

