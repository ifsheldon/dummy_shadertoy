use num_traits::Pow;
use pixel_canvas::Color;

use crate::data::{ScalarMul, Vec3, Vec4, Mat4};
use crate::shading::{Object, ShapeTypes, Light};

#[derive(Debug, Copy, Clone)]
pub struct EMA {
    pre: f32,
    alpha: f32,
    beta: f32,
    enable_correction: bool,
    t: u8,
}

impl EMA {
    pub fn new(alpha: f32, enable_correction: bool) -> Self {
        EMA {
            pre: 0.0,
            alpha,
            beta: 1.0 - alpha,
            enable_correction,
            t: 0,
        }
    }

    pub fn set_alpha(&mut self, alpha: f32) {
        self.alpha = alpha;
        self.beta = 1.0 - alpha;
    }
    pub fn add_stat(&mut self, v: f32) {
        self.pre = self.alpha * self.pre + self.beta * v;
        self.t += 1;
    }
    pub fn get(&self) -> f32 {
        return if self.enable_correction && self.t != 0 && self.t < 200 {
            self.pre / (1.0 - self.alpha.pow(self.t))
        } else {
            self.pre
        };
    }

    pub fn clear(&mut self) {
        self.pre = 0.0;
        self.t = 0;
    }
}

#[inline]
pub fn to_color(mut color: Vec3) -> Color {
    clamp_(&mut color);
    color.scalar_mul_(255.);
    let x = color.r().round();
    let y = color.g().round();
    let z = color.b().round();
    Color::rgb(x as u8, y as u8, z as u8)
}

#[inline]
fn clamp_(color: &mut Vec3) {
    color.set_r(clamp_float(color.r()));
    color.set_g(clamp_float(color.g()));
    color.set_b(clamp_float(color.b()));
}

#[inline]
fn clamp_float(x: f32) -> f32 {
    if x < 0. {
        return 0.;
    }
    if x > 1. {
        return 1.;
    }
    x
}

pub fn add_plane(
    objects: &mut Vec<Object>,
    coefficients: &Vec4,
    material_id: usize,
    transformation: Mat4,
) {
    let o = Object {
        shape: ShapeTypes::Plane(
            coefficients.x(),
            coefficients.y(),
            coefficients.z(),
            coefficients.w(),
        ),
        original_transformation: transformation.clone(),
        transformation,
        material_id,
    };
    objects.push(o);
}

pub fn add_sphere(
    objects: &mut Vec<Object>,
    radius: f32,
    material_id: usize,
    transformation: Mat4,
) {
    let o = Object {
        transformation,
        original_transformation: transformation.clone(),
        shape: ShapeTypes::Sphere(radius),
        material_id,
    };
    objects.push(o);
}

pub fn add_ellipsoid(
    objects: &mut Vec<Object>,
    dimensions: Vec3,
    material_id: usize,
    transformation: Mat4,
) {
    let o = Object {
        transformation,
        original_transformation: transformation.clone(),
        shape: ShapeTypes::Ellipsoid(dimensions.x(), dimensions.y(), dimensions.z()),
        material_id,
    };
    objects.push(o);
}

pub fn add_rounded_cylinder(
    objects: &mut Vec<Object>,
    radius: f32,
    round_radius: f32,
    height: f32,
    material_id: usize,
    transformation: Mat4,
) {
    let o = Object {
        transformation,
        original_transformation: transformation.clone(),
        shape: ShapeTypes::RoundedCylinder(radius, round_radius, height),
        material_id,
    };
    objects.push(o);
}


pub fn add_light(lights: &mut Vec<Light>, position: Vec3, ambient: Vec3, source: Vec3) {
    let l = Light {
        position,
        original_position: position.clone(),
        ambient,
        diffuse: source,
        r: 0.1,
    };
    lights.push(l);
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_EMA() {
        let series = [1.0, 1.0, 2.0, 3.0, 2.0, 1.0, 4.0, 5.0, 6.0, 8.0];
        let mut ema = EMA::new(0.99, true);
        for i in series.iter() {
            ema.add_stat(*i);
            println!("{}", ema.get());
        }
    }
}