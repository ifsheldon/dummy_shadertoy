// Code Reference:
// * My fragment shader code that was written in the NUS CS Workshop of Real Time Rendering
//      hosted on Shadertoy https://www.shadertoy.com/view/wlsSzs
// * Analytical formulas by Inigo Quilez, source: http://iquilezles.org/www/articles/distfunctions/distfunctions.htm
// * glm source code on https://github.com/g-truc/glm

use std::ops::{Index, IndexMut};
use std::time::Instant;

use pixel_canvas::{Canvas, Color, XY};
use pixel_canvas::input::glutin::event::VirtualKeyCode;
use pixel_canvas::input::glutin::event::VirtualKeyCode::W;
use rand::prelude::*;
use rayon::prelude::*;

use crate::data::{Add, Length, Mat4, Minus, Normalize, ScalarDiv, ScalarMul, Vec3, Vec4};
use crate::shading::*;
use crate::shapes::{
    sdf_cube, sdf_cylinder, sdf_ellipsoid, sdf_plane, sdf_rounded_cylinder, sdf_sphere,
};
use crate::state::KeyboardMouseStates;
use crate::tex::{Interpolation, Tex2D, Tiling};
use crate::transformations::*;
use crate::utils::*;

mod data;
mod err;
mod shading;
mod shapes;
mod state;
mod transformations;
mod tex;
mod utils;

const EPSILON: f32 = 0.0001;
const MIN_DIST: f32 = 0.0;
const MAX_DIST: f32 = 100.0;
const MAX_MARCHING_STEPS: i32 = 255;
const NUM_ITERATIONS: i32 = 2;
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
        r: 0.1,
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
    Zoom,
    AutoMoveCam,
}


pub struct Pixel {
    pub x: usize,
    pub y: usize,
    pub x_f: f32,
    pub y_f: f32,
    ema_r: EMA,
    ema_g: EMA,
    ema_b: EMA,
}

impl Pixel {
    pub fn new_ema_pixel(x: usize, y: usize, alpha: f32) -> Self {
        Pixel {
            x,
            y,
            x_f: x as f32,
            y_f: y as f32,
            ema_r: EMA::new(alpha, true),
            ema_g: EMA::new(alpha, true),
            ema_b: EMA::new(alpha, true),
        }
    }

    pub fn update_color(&mut self, color_f: &Vec3) {
        self.ema_r.add_stat(color_f.r());
        self.ema_g.add_stat(color_f.g());
        self.ema_b.add_stat(color_f.b());
    }

    pub fn get_color_f(&self) -> Vec3 {
        Vec3::new_rgb(self.ema_r.get(), self.ema_g.get(), self.ema_b.get())
    }

    pub fn get_color_u8(&self) -> Color {
        to_color(Vec3::new_rgb(
            self.ema_r.get(),
            self.ema_g.get(),
            self.ema_b.get(),
        ))
    }

    pub fn clear_color(&mut self) {
        self.ema_r.clear();
        self.ema_g.clear();
        self.ema_b.clear();
    }
}

fn main() {
    const VIEW_PLANE_WIDTH: f32 = 4.;
    const VIEW_PLANE_HEIGHT: f32 = 3.;
    const SUPER_SAMPLE_RATE: usize = 2; // super sample cost = super sample rate ^ 2
    const SUPER_SAMPLE_RATE_F: f32 = SUPER_SAMPLE_RATE as f32;

    let tex = Tex2D::from_file(String::from("./tex.jpg"), Interpolation::Bilinear, Tiling::Repeat).expect("What happened?");
    let mut super_sample_indices = Vec::new();
    for x in 0..SUPER_SAMPLE_RATE {
        for y in 0..SUPER_SAMPLE_RATE {
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
    let mut enable_super_sample = false;
    let mut enable_motion_blur = false;
    let mut enable_soft_shadow = false;
    let mut enable_dov = false;
    let mut enable_glossy = false;
    let mut enable_env_mapping = false;
    let original_focus_plane_to_eye_dist = 0.5;
    let mut focus_plane_to_eye_dist = original_focus_plane_to_eye_dist;
    let dov_eye_width_wc = 0.5;
    let dov_eye_height_wc = 0.5;
    let mut doved = false;
    let soft_shadow_pass_num = 5;
    let mut pass_num = 0;
    let mut clear_before_drawing = false;
    let mut center = Vec3::new(0.);
    let center_original = center.clone();
    let up = Vec3::new_xyz(0.0, 1.0, 0.0);

    let now = Instant::now();
    // configure the window/canvas
    let avg_last_frame_num = 5.0;
    let alpha = 1.0 - 1.0 / avg_last_frame_num;
    let mut pixels = Vec::new();
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            pixels.push(Pixel::new_ema_pixel(x, y, alpha));
        }
    }
    let canvas = Canvas::new(WIDTH, HEIGHT)
        .title("Dynamic Raytracer")
        .state(KeyboardMouseStates::new())
        .input(KeyboardMouseStates::handle_input);

    let mut render_time_ema = EMA::new(0.95, true);

    let mut mode = Mode::Orbit;

    let mut theta: f32 = 0.; // wc the angle between -z and +x
    let mut phi: f32 = std::f32::consts::FRAC_PI_2; // wc complement angle of the angle between the line and z-x plane

    let mut auto_moving_angle: f32 = 0.;
    let angle_delta = (2.5_f32).to_radians();

    // render up to 60fps
    canvas.render(move |state, frame_buffer_image| {
        let before = now.elapsed().as_millis();
        // switching modes
        let mut switching_mode = true;
        if state.received_keycode {
            match state.keycode {
                VirtualKeyCode::Key9 =>{
                    enable_env_mapping = true;
                    println!("Enabled Sphere Env Mapping");
                    eye_changed = true;
                    clear_before_drawing = true;
                }
                VirtualKeyCode::Key0 =>{
                    enable_env_mapping = false;
                    println!("Disabled Sphere Env Mapping");
                    eye_changed = true;
                    clear_before_drawing = true;
                }
                VirtualKeyCode::G => {
                    enable_glossy = true;
                    println!("Enable Glossy");
                    eye_changed = true;
                    clear_before_drawing = true;
                }
                VirtualKeyCode::H => {
                    enable_glossy = false;
                    println!("Disabled Glossy");
                    eye_changed = true;
                    clear_before_drawing = true;
                }
                VirtualKeyCode::Equals => {
                    enable_super_sample = true;
                    println!("Enabled Super Sample");
                    eye_changed = true;
                }
                VirtualKeyCode::Subtract | VirtualKeyCode::Minus => {
                    enable_super_sample = false;
                    println!("Disabled Super Sample");
                    eye_changed = true;
                    clear_before_drawing = true;
                }
                VirtualKeyCode::Left => {
                    enable_dov = false;
                    println!("Disabled DOV");
                    eye_changed = true;
                    clear_before_drawing = true;
                    focus_plane_to_eye_dist = original_focus_plane_to_eye_dist;
                }
                VirtualKeyCode::Right => {
                    enable_dov = true;
                    println!("Enabled DOV");
                    eye_changed = true;
                    enable_dov = true;
                }
                VirtualKeyCode::Up => {
                    if enable_dov {
                        focus_plane_to_eye_dist += 0.1;
                        doved = false;
                        println!("Focus Plane to eye distance = {}", focus_plane_to_eye_dist);
                    }
                }
                VirtualKeyCode::Down => {
                    if enable_dov {
                        focus_plane_to_eye_dist -= 0.1;
                        if focus_plane_to_eye_dist <= 0.1 {
                            focus_plane_to_eye_dist = 0.1;
                        }
                        println!("Focus Plane to eye distance = {}", focus_plane_to_eye_dist);
                        doved = false;
                    }
                }
                VirtualKeyCode::J => {
                    enable_soft_shadow = false;
                    println!("Disabled Soft Shadow");
                    eye_changed = true;
                    clear_before_drawing = true;
                }
                VirtualKeyCode::K => {
                    enable_soft_shadow = true;
                    println!("Enabled Soft Shadow");
                    eye_changed = true;
                    clear_before_drawing = true;
                }
                VirtualKeyCode::M => {
                    enable_motion_blur = true;
                    enable_super_sample = false;
                    println!("Enabled Motion Blur, Disabled Super Sample");
                }
                VirtualKeyCode::N => {
                    enable_motion_blur = false;
                    clear_before_drawing = true;
                    println!("Disabled Motion Blur");
                }
                VirtualKeyCode::Key1 => {
                    mode = Mode::Orbit;
                    clear_before_drawing = true;
                    println!("Chose Mode: {:?}", mode);
                }
                VirtualKeyCode::Key4 => {
                    mode = Mode::AutoMoveCam;
                    println!("Chose Mode: {:?}", mode);
                }
                VirtualKeyCode::Z => {
                    mode = Mode::Zoom;
                    println!("Chose Mode: {:?}", mode);
                }
                _ => switching_mode = false,
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
                _ => eye_changed = false,
            }
            let radius = eye_pos._minus(&center).get_length();
            eye_pos.set_y(phi.cos() * radius);
            eye_pos.set_z(-theta.cos() * radius * phi.sin());
            eye_pos.set_x(theta.sin() * radius * phi.sin());
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
                _ => eye_changed = false,
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
        if eye_changed || enable_motion_blur || clear_before_drawing {
            pixels.par_iter_mut().for_each(|pixel| {
                let frag_coord = [pixel.x_f, pixel.y_f];
                let primary_ray =
                    get_ray_perspective(fov_radian, &look_at_mat, &eye_pos, &frag_coord);
                if clear_before_drawing || !enable_motion_blur {
                    pixel.clear_color();
                }
                pixel.update_color(&shade(primary_ray, &objects, &materials, &lights, false, enable_glossy, enable_env_mapping, &tex));
            });
            super_sampled = false;
            doved = false;
            rendered = true;
            pass_num = 0;
            if clear_before_drawing {
                clear_before_drawing = false; // reset the flag
            }
        } else if (enable_super_sample && !super_sampled) || (enable_soft_shadow && pass_num < soft_shadow_pass_num) {
            let grid_size = 1.0 / SUPER_SAMPLE_RATE_F;
            pixels.par_iter_mut().for_each(|pixel| {
                let rand_colors: Vec<Vec3> = super_sample_indices
                    .par_iter()
                    .map(|idx| {
                        let mut random_generator = rand::thread_rng();
                        let grid_x = idx.0;
                        let grid_y = idx.1;
                        let grid_base_x = pixel.x_f + grid_x as f32 * grid_size;
                        let grid_base_y = pixel.y_f + grid_y as f32 * grid_size;
                        let rand_x = grid_base_x + random_generator.gen_range(0.0, grid_size);
                        let rand_y = grid_base_y + random_generator.gen_range(0.0, grid_size);
                        let frag_coord = [rand_x, rand_y];
                        let rand_ray =
                            get_ray_perspective(fov_radian, &look_at_mat, &eye_pos, &frag_coord);
                        let color_f = shade(rand_ray, &objects, &materials, &lights, enable_soft_shadow, false, enable_env_mapping, &tex);
                        return color_f;
                    })
                    .collect();
                rand_colors
                    .iter()
                    .for_each(|color| pixel.update_color(color));
            });
            super_sampled = true;
            rendered = true;
            if enable_soft_shadow {
                pass_num += 1;
            }
        } else if enable_dov && !doved {
            doved = true;
            let grid_x = dov_eye_width_wc / SUPER_SAMPLE_RATE_F;
            let grid_y = dov_eye_width_wc / SUPER_SAMPLE_RATE_F;
            let off_x = -0.5 * dov_eye_width_wc;
            let off_y = -0.5 * dov_eye_height_wc;
            pixels.par_iter_mut().for_each(|pixel| {
                let frag_coord = [pixel.x_f, pixel.y_f];
                let test_ray =
                    get_ray_perspective(fov_radian, &look_at_mat, &eye_pos, &frag_coord);
                let focus_point_wc = test_ray.origin._add(&test_ray.direction.scalar_mul(focus_plane_to_eye_dist));
                let mut camera_up = look_at_mat._get_column(1);
                let mut camera_right = look_at_mat._get_column(0);
                let rand_colors: Vec<Vec3> = super_sample_indices.par_iter().map(|idx| {
                    let mut random_generator = rand::thread_rng();
                    let jitter_x = random_generator.gen_range(0.0, grid_x);
                    let jitter_y = random_generator.gen_range(0.0, grid_y);
                    let base_x = idx.0 as f32 * grid_x;
                    let base_y = idx.1 as f32 * grid_y;
                    let r = jitter_x + base_x + off_x;
                    let u = jitter_y + base_y + off_y;
                    let up = camera_up.scalar_mul(u);
                    let right = camera_right.scalar_mul(r);
                    let mut cam_pos = eye_pos._add(&right);
                    cam_pos.add_(&up);
                    let mut dir = focus_point_wc._minus(&cam_pos);
                    dir.normalize_();
                    let ray = Ray {
                        origin: cam_pos,
                        direction: dir,
                    };
                    let color_f = shade(ray, &objects, &materials, &lights, false, false, enable_env_mapping, &tex);
                    return color_f;
                }).collect();
                pixel.clear_color();
                rand_colors
                    .iter()
                    .for_each(|color| pixel.update_color(color));
            });
            rendered = true;
        }
        if rendered {
            frame_buffer_image
                .par_iter_mut()
                .enumerate()
                .for_each(|(idx, pixel)| {
                    let pix: &Pixel = pixels.get(idx).unwrap();
                    *pixel = pix.get_color_u8();
                });
            let after = now.elapsed().as_millis();
            let t = after - before;
            if super_sampled {
                println!(
                    "Took {} ms to super-sample one frame, super-sample rate = {}X",
                    t, SUPER_SAMPLE_RATE
                );
            } else {
                println!("Took {} ms to render one frame", t);
            }
            render_time_ema.add_stat(t as f32);
            println!("Render time EMA = {}", render_time_ema.get());
        }
        state.reset_flags();
        eye_changed = false;
    });
}

// fn main()
// {
//     const WIDTH: usize = 512;
//     const HEIGHT: usize = 512;
//     let canvas = Canvas::new(WIDTH, HEIGHT).title("Textrue Test");
//     let tex = Tex2D::from_file(String::from("./tex.jpg"), Interpolation::Bilinear, Tiling::Repeat).expect("What happened?");
//     let mut rendered = false;
//     canvas.render(move |_, frame_buf| {
//         frame_buf.par_iter_mut().enumerate().for_each(|(idx, color)| {
//             let y = idx / WIDTH;
//             let x = idx % WIDTH;
//             let u = x as f32 / WIDTH as f32;
//             let v = y as f32 / HEIGHT as f32;
//             let rgba = tex.get_color_u8(u, v);
//             if rendered { return; }
//             let c = Color {
//                 r: rgba.0[0],
//                 g: rgba.0[1],
//                 b: rgba.0[2],
//             };
//             *color = c;
//         });
//         rendered = true;
//     })
// }