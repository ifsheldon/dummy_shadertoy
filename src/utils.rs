use num_traits::Pow;
use pixel_canvas::Color;

use crate::data::{ScalarMul, Vec3};

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