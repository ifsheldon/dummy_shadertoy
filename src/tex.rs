use std::fs::File;
use std::io;
use std::io::{BufReader, Error};
use std::mem::take;

use image::{DynamicImage, GenericImageView, ImageError, Rgb, Rgba};
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
        unimplemented!()
    }

    pub fn get_color_u8(&self, u: f32, v: f32) -> Rgba<u8> {
        let (u, v) = self.check_uv(u, v);
        let x = u * (self.width_f - 1.0);
        let y = v * (self.height_f - 1.0);
        return match self.interpolation {
            Interpolation::Bilinear => {
                let x_lower = x.floor() as u32;
                let x_upper = x_lower + 1;
                let x_frac = x.fract();
                let y_lower = y.floor() as u32;
                let y_upper = y_lower + 1;
                let y_frac = y.fract();
                let x0y0 = self.img.get_pixel(x_lower, y_lower);
                let x0y1 = self.img.get_pixel(x_lower, y_upper);
                let x1y0 = self.img.get_pixel(x_upper, y_lower);
                let x1y1 = self.img.get_pixel(x_upper, y_upper);
                let interpolated = Self::biliniear_interpolate_color(&x0y0, &x1y0, &x0y1, &x1y1, x_frac, y_frac);
                interpolated
            }
            Interpolation::Nearest => {
                let x_round = x.round();
                let y_round = y.round();
                let idx_x = x_round as u32;
                let idx_y = y_round as u32;
                self.img.get_pixel(idx_x, idx_y)
            }
        }
    }

    fn biliniear_interpolate_color(x0y0: &Rgba<u8>, x1y0: &Rgba<u8>, x0y1: &Rgba<u8>, x1y1: &Rgba<u8>, x_frac: f32, y_frac: f32) -> Rgba<u8> {
        let (x0y0_r, x0y0_g, x0y0_b, x0y0_a) = Self::take_rgba(x0y0);
        let (x1y0_r, x1y0_g, x1y0_b, x1y0_a) = Self::take_rgba(x1y0);
        let (x0y1_r, x0y1_g, x0y1_b, x0y1_a) = Self::take_rgba(x0y1);
        let (x1y1_r, x1y1_g, x1y1_b, x1y1_a) = Self::take_rgba(x1y1);
        let r = Self::bilinear_interpolate(x0y0_r, x1y0_r, x0y1_r, x1y1_r, x_frac, y_frac);
        let g = Self::bilinear_interpolate(x0y0_g, x1y0_g, x0y1_g, x1y1_g, x_frac, y_frac);
        let b = Self::bilinear_interpolate(x0y0_b, x1y0_b, x0y1_b, x1y1_b, x_frac, y_frac);
        let a = Self::bilinear_interpolate(x0y0_a, x1y0_a, x0y1_a, x1y1_a, x_frac, y_frac);
        return Rgba([Self::clamp(r) as u8, Self::clamp(g) as u8, Self::clamp(b) as u8, Self::clamp(a) as u8]);
    }

    fn clamp(v: f32) -> f32 {
        let mut v = v;
        if v > 255.0 {
            v = 255.0;
        } else if v < 0.0 {
            v = 0.0;
        }
        return v;
    }
    fn take_rgba(color: &Rgba<u8>) -> (f32, f32, f32, f32)
    {
        (Self::take_r(color), Self::take_g(color), Self::take_b(color), Self::take_a(color))
    }

    #[inline]
    fn take_r(color: &Rgba<u8>) -> f32 {
        color.0[0] as f32
    }

    #[inline]
    fn take_g(color: &Rgba<u8>) -> f32 {
        color.0[1] as f32
    }

    #[inline]
    fn take_b(color: &Rgba<u8>) -> f32 {
        color.0[2] as f32
    }

    #[inline]
    fn take_a(color: &Rgba<u8>) -> f32 {
        color.0[3] as f32
    }

    fn bilinear_interpolate(x0y0: f32, x1y0: f32, x0y1: f32, x1y1: f32, x_frac: f32, y_frac: f32) -> f32 {
        let _x_frac = 1.0 - x_frac;
        let _y_frac = 1.0 - y_frac;
        let y0 = Self::mix(x0y0, x1y0, _x_frac);
        let y1 = Self::mix(x0y1, x1y1, _x_frac);
        return Self::mix(y0, y1, _y_frac);
    }

    #[inline]
    fn mix(a: f32, b: f32, alpha: f32) -> f32 {
        let beta = 1.0 - alpha;
        return a * alpha + b * beta;
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

