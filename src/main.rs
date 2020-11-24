// Code Reference:
// * My fragment shader code that was written in the NUS CS Workshop of Real Time Rendering
//      hosted on Shadertoy https://www.shadertoy.com/view/wlsSzs
// * Analytical formulas by Inigo Quilez, source: http://iquilezles.org/www/articles/distfunctions/distfunctions.htm
// * glm source code on https://github.com/g-truc/glm

use std::time::Instant;

use num_traits::Pow;
use pixel_canvas::{Canvas, Color};
use pixel_canvas::input::glutin::event::VirtualKeyCode;
use rand::prelude::*;
use rayon::prelude::*;

use crate::data::{Add, Length, Mat4, Minus, ScalarDiv, ScalarMul, Vec3, Vec4};
use crate::shading::*;
use crate::shapes::{
    sdf_cube, sdf_cylinder, sdf_ellipsoid, sdf_plane, sdf_rounded_cylinder, sdf_sphere,
};
use crate::state::KeyboardMouseStates;
use crate::transformations::*;

mod data;
mod err;
mod shading;
mod shapes;
mod state;
mod transformations;

const EPSILON: f32 = 0.0001;
const MIN_DIST: f32 = 0.0;
const MAX_DIST: f32 = 100.0;
const MAX_MARCHING_STEPS: i32 = 255;
const NUM_ITERATIONS: i32 = 3;
const BACKGROUND_COLOR: (f32, f32, f32) = (0.4, 0.4, 0.4);
const WIDTH: usize = 640;
const WIDTH_F: f32 = WIDTH as f32;
const WIDTH_HF: f32 = WIDTH_F / 2.;
const HEIGHT: usize = 480;
const HEIGHT_F: f32 = HEIGHT as f32;
const HEIGHT_HF: f32 = HEIGHT_F / 2.;

pub fn add_cube(
    objects: &mut Vec<Object>,
    width: f32,
    height: f32,
    depth: f32,
    material_id: usize,
    transformation: Mat4,
) {
    let o = Object {
        shape: ShapeTypes::Cube(width, height, depth),
        original_transformation: transformation.clone(),
        transformation,
        material_id,
    };
    objects.push(o);
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

pub fn add_cylinder(
    objects: &mut Vec<Object>,
    radius: f32,
    height: f32,
    material_id: usize,
    transformation: Mat4,
) {
    let o = Object {
        transformation,
        original_transformation: transformation.clone(),
        shape: ShapeTypes::Cylinder(radius, height),
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
    };
    lights.push(l);
}

pub fn init_scene(
    objects: &mut Vec<Object>,
    materials: &mut Vec<Material>,
    lights: &mut Vec<Light>,
) {
    let material_gray = Material {
        diffuse: Vec3::new(0.5),
        ambient: Vec3::new(0.1),
        reflection: Vec3::new(1.0),
        global_reflection: Vec3::new(0.5),
        specular: 64.0,
    };
    let material_red = Material {
        diffuse: Vec3::new_rgb(1., 0., 0.),
        ambient: Vec3::new_rgb(1., 0., 0.),
        reflection: Vec3::new(1.),
        global_reflection: Vec3::new(0.2),
        specular: 10.,
    };
    let material_green = Material {
        diffuse: Vec3::new_rgb(0., 1., 0.),
        ambient: Vec3::new_rgb(0., 1., 0.),
        reflection: Vec3::new(1.),
        global_reflection: Vec3::new(0.2),
        specular: 10.,
    };
    let material_blue = Material {
        diffuse: Vec3::new_rgb(0., 0., 1.),
        ambient: Vec3::new_rgb(0., 0., 1.),
        reflection: Vec3::new(1.),
        global_reflection: Vec3::new(0.1),
        specular: 10.,
    };

    materials.push(material_gray); //idx =0
    materials.push(material_red); //idx =1
    materials.push(material_green); //idx =2
    materials.push(material_blue); //idx =3

    let identity = Mat4::identity();
    // red sphere
    let transformation = translate_obj(identity.clone(), &Vec3::new_xyz(1., 1., 0.5));
    add_sphere(objects, 1., 1, transformation);

    //gray plane
    let transformation = translate_obj(identity.clone(), &Vec3::new_xyz(0., -1.4, 0.));
    add_plane(objects, &Vec4::new_xyzw(0., 1.0, 0., 0.), 0, transformation);

    //green ellipsoid
    let transformation = translate_obj(identity.clone(), &Vec3::new_xyz(-0.1, 0.0, 0.4));
    add_ellipsoid(objects, Vec3::new_xyz(0.4, 0.2, 0.4), 2, transformation);

    // blue rounded cylinder
    let transformation = translate_obj(identity.clone(), &Vec3::new_xyz(-0.5, -0.3, -0.1));
    // add_cylinder(objects, 0.1, 0.1, 3, transformation);
    add_rounded_cylinder(objects, 0.1, 0.02, 0.3, 3, transformation);

    //white light
    add_light(
        lights,
        Vec3::new_xyz(0., 5., 0.),
        Vec3::new(0.3),
        Vec3::new(0.7),
    );
}

pub fn cast_hit_ray(ray: &Ray, objects: &Vec<Object>) -> Option<(i32, Vec3)> {
    let (obj_idx, dist) = shortest_dist_to_surface(objects, &ray.origin, &ray.direction, -1);
    return if dist > MAX_DIST - EPSILON {
        None
    } else {
        let hit_position = ray.origin._add(&ray.direction.scalar_mul(dist));
        Some((obj_idx, hit_position))
    };
}

#[derive(PartialEq, Debug)]
pub enum Mode {
    Orbit,
    FreeMove,
    Zoom,
    MovingLight,
    AutoMoveCam,
}

pub struct EMA
{
    pre: f32,
    alpha: f32,
    beta: f32,
    enable_correction: bool,
    t: u8,
}

impl EMA
{
    pub fn new(alpha: f32, enable_correction: bool) -> Self
    {
        EMA {
            pre: 0.0,
            alpha,
            beta: 1.0 - alpha,
            enable_correction,
            t: 0,
        }
    }

    pub fn set_alpha(&mut self, alpha: f32)
    {
        self.alpha = alpha;
        self.beta = 1.0 - alpha;
    }
    pub fn add_stat(&mut self, v: f32)
    {
        self.pre = self.alpha * self.pre + self.beta * v;
        self.t += 1;
    }
    pub fn get(&self) -> f32
    {
        return if self.enable_correction && self.t != 0 && self.t < 200
        {
            self.pre / (1.0 - self.alpha.pow(self.t))
        } else {
            self.pre
        };
    }

    pub fn clear(&mut self)
    {
        self.pre = 0.0;
        self.t = 0;
    }
}

fn main() {
    const VIEW_PLANE_WIDTH: f32 = 4.;
    const VIEW_PLANE_HEIGHT: f32 = 3.;
    const SUPER_SAMPLE_RATE: usize = 2; // super sample cost = super sample rate ^ 2
    const SUPER_SAMPLE_RATE_F: f32 = SUPER_SAMPLE_RATE as f32;

    let mut super_sample_indices = Vec::new();
    for x in 0..SUPER_SAMPLE_RATE {
        for y in 0..SUPER_SAMPLE_RATE
        {
            super_sample_indices.push((x, y));
        }
    }

    let mut objects = Vec::new();
    let mut lights = Vec::new();
    let mut materials = Vec::new();
    init_scene(&mut objects, &mut materials, &mut lights);
    let fov_radian = (2.0_f32).atan() * 2.;

    let mut eye_pos = Vec3::new_xyz(0.0, 0.0, -1.0);
    let mut eye_changed = true; //for the first frame
    let mut super_sampled = false;
    let eye_pos_original = eye_pos.clone();
    let mut center = Vec3::new(0.);
    let center_original = center.clone();
    let up = Vec3::new_xyz(0.0, 1.0, 0.0);

    let dw = VIEW_PLANE_WIDTH / WIDTH_F;
    let dh = VIEW_PLANE_HEIGHT / HEIGHT_F;

    let now = Instant::now();
    // configure the window/canvas
    let canvas = Canvas::new(WIDTH, HEIGHT)
        .title("Dynamic Raytracer")
        .state(KeyboardMouseStates::new())
        .input(KeyboardMouseStates::handle_input);

    let mut render_time_ema = EMA::new(0.95, false);

    let mut mode = Mode::Orbit;

    let mut theta: f32 = 0.; // wc the angle between -z and +x
    let mut phi: f32 = std::f32::consts::FRAC_PI_2; // wc complement angle of the angle between the line and z-x plane

    let mut auto_moving_angle: f32 = 0.;
    let angle_delta = (2.5_f32).to_radians();

    // render up to 60fps
    canvas.render(move |state, frame_buffer_image| {
        let before = now.elapsed().as_millis();
        // switching modes
        let mut switching_mode = false;
        if state.received_keycode {
            match state.keycode {
                VirtualKeyCode::Key1 => {
                    mode = Mode::Orbit;
                    println!("Chose Mode: {:?}", mode);
                    switching_mode = true;
                }
                VirtualKeyCode::Key3 => {
                    mode = Mode::FreeMove;
                    println!("Chose Mode: {:?}", mode);
                    switching_mode = true
                }
                VirtualKeyCode::Key4 => {
                    mode = Mode::AutoMoveCam;
                    println!("Chose Mode: {:?}", mode);
                    switching_mode = true;
                }
                VirtualKeyCode::Z => {
                    mode = Mode::Zoom;
                    println!("Chose Mode: {:?}", mode);
                    switching_mode = true
                }
                VirtualKeyCode::C => {
                    switching_mode = true;
                }
                VirtualKeyCode::V => {
                    mode = Mode::MovingLight;
                    println!("Chose Mode: {:?}", mode);
                    switching_mode = true;
                }
                _ => {}
            }
        }
        // moving the light
        if !switching_mode && state.received_keycode && mode == Mode::MovingLight {
            let light = lights.get_mut(0).unwrap();
            let light_pos = &mut light.position;
            match state.keycode {
                VirtualKeyCode::A => light_pos.set_x(light_pos.x() - 0.1),
                VirtualKeyCode::D => light_pos.set_x(light_pos.x() + 0.1),
                VirtualKeyCode::W => light_pos.set_z(light_pos.z() + 0.1),
                VirtualKeyCode::S => light_pos.set_z(light_pos.z() - 0.1),
                VirtualKeyCode::Q => light_pos.set_y(light_pos.y() + 0.1),
                VirtualKeyCode::E => light_pos.set_y(light_pos.y() - 0.1),
                VirtualKeyCode::R => *light_pos = light.original_position.clone(),
                _ => {}
            }
        }
        // Orbiting camera
        if !switching_mode && state.received_keycode && mode == Mode::Orbit {
            println!("Key Pressed: {:?}", state.keycode);
            match state.keycode {
                // for orbiting around the origin
                VirtualKeyCode::A => {
                    theta += (5.0_f32).to_radians();
                    eye_changed = true
                }
                VirtualKeyCode::D => {
                    theta -= (5.0_f32).to_radians();
                    eye_changed = true
                }
                VirtualKeyCode::W => {
                    phi -= (2.5_f32).to_radians();
                    eye_changed = true;
                    if phi <= 0. {
                        phi = 0.01;
                        eye_changed = false;
                    }
                }
                VirtualKeyCode::S => {
                    phi += (2.5_f32).to_radians();
                    eye_changed = true;
                    if phi >= std::f32::consts::PI {
                        eye_changed = false;
                        phi = std::f32::consts::PI - 0.01;
                    }
                }
                VirtualKeyCode::R => {
                    theta = 0.;
                    phi = std::f32::consts::FRAC_PI_2;
                    center = center_original.clone();
                    eye_changed = true;
                }
                _ => eye_changed = false
            }
            let radius = eye_pos._minus(&center).get_length();
            eye_pos.set_y(phi.cos() * radius);
            eye_pos.set_z(-theta.cos() * radius * phi.sin());
            eye_pos.set_x(theta.sin() * radius * phi.sin());
        }
        // Moving camera in wc
        if !switching_mode && state.received_keycode && mode == Mode::FreeMove {
            match state.keycode {
                VirtualKeyCode::A => eye_pos.set_x(eye_pos.x() - 0.1),
                VirtualKeyCode::D => eye_pos.set_x(eye_pos.x() + 0.1),
                VirtualKeyCode::W => eye_pos.set_z(eye_pos.z() + 0.1),
                VirtualKeyCode::S => eye_pos.set_z(eye_pos.z() - 0.1),
                VirtualKeyCode::Q => eye_pos.set_y(eye_pos.y() + 0.1),
                VirtualKeyCode::E => eye_pos.set_y(eye_pos.y() - 0.1),
                VirtualKeyCode::R => {
                    eye_pos = eye_pos_original.clone();
                    center = center_original.clone();
                }
                _ => {}
            }
        }
        let look_at_mat = look_at(&eye_pos, &center, &up);
        // zooming
        if !switching_mode && state.received_keycode && mode == Mode::Zoom {
            let mut camera_focus_direction = look_at_mat._get_column(2);
            match state.keycode {
                VirtualKeyCode::Q => {
                    camera_focus_direction.scalar_mul_(0.1);
                    eye_pos.add_(&camera_focus_direction);
                    eye_changed = true;
                }
                VirtualKeyCode::E => {
                    camera_focus_direction.scalar_mul_(0.1);
                    eye_pos.minus_(&camera_focus_direction);
                    eye_changed = true;
                }
                _ => eye_changed = false
            }
        }
        // for automatic moving the camera
        if mode == Mode::AutoMoveCam {
            eye_pos.set_x(auto_moving_angle.sin());
            eye_pos.set_z(-auto_moving_angle.cos());
            eye_pos.set_y(0.);
            auto_moving_angle += angle_delta;
            eye_changed = true;
        }
        // multi-pass render
        let mut rendered = false;
        if eye_changed {
            frame_buffer_image
                .par_iter_mut()
                .enumerate()
                .for_each(|(idx, pixel)| {
                    let y = idx / WIDTH;
                    let x = idx % WIDTH;
                    let frag_coord = [x as f32, y as f32];
                    let primary_ray = get_ray_perspective(fov_radian, &look_at_mat, &eye_pos, &frag_coord);
                    *pixel = to_color(shade(primary_ray, &objects, &materials, &lights));
                });
            super_sampled = false;
            rendered = true;
        } else if !super_sampled {
            frame_buffer_image
                .par_iter_mut()
                .enumerate()
                .for_each(|(idx, pixel)| {
                    let y = idx / WIDTH;
                    let x = idx % WIDTH;
                    let grid_size = 1.0 / SUPER_SAMPLE_RATE_F;
                    // should shoot more rays
                    let rand_colors: Vec<Vec3> = super_sample_indices.par_iter().map(|idx| {
                        let mut random_generator = rand::thread_rng();
                        let grid_x = idx.0;
                        let grid_y = idx.1;
                        let grid_base_x = x as f32 + grid_x as f32 * grid_size;
                        let grid_base_y = y as f32 + grid_y as f32 * grid_size;
                        let rand_x = grid_base_x + random_generator.gen_range(0.0, grid_size);
                        let rand_y = grid_base_y + random_generator.gen_range(0.0, grid_size);
                        let frag_coord = [rand_x, rand_y];
                        let rand_ray = get_ray_perspective(fov_radian, &look_at_mat, &eye_pos, &frag_coord);
                        let color_f = shade(rand_ray, &objects, &materials, &lights);
                        return color_f;
                    }).collect();
                    let mut color_sum = Vec3::new_rgb((pixel.r as f32) / 255.0, (pixel.g as f32) / 255.0, (pixel.b as f32) / 255.0);
                    for c in rand_colors.iter()
                    {
                        color_sum.add_(c);
                    }
                    color_sum.scalar_div_((SUPER_SAMPLE_RATE * SUPER_SAMPLE_RATE + 1) as f32);
                    *pixel = to_color(color_sum);
                });
            super_sampled = true;
            rendered = true;
        }else {
            // do nothing
        }
        if rendered
        {
            let after = now.elapsed().as_millis();
            let t = after - before;
            println!("Took {} ms to render one frame", t);
            render_time_ema.add_stat(t as f32);
            println!("Render time EMA = {}", render_time_ema.get());
        }
        state.reset_flags();
        eye_changed = false;
    });
}

#[inline]
fn to_color(mut color: Vec3) -> Color {
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

#[cfg(test)]
mod test
{
    use super::*;

    #[test]
    fn test_EMA()
    {
        let series = [1.0, 1.0, 2.0, 3.0, 2.0, 1.0, 4.0, 5.0, 6.0, 8.0];
        let mut ema = EMA::new(0.99, true);
        for i in series.iter()
        {
            ema.add_stat(*i);
            println!("{}", ema.get());
        }
    }
}