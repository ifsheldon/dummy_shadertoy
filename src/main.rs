// Code Reference:
// * My fragment shader code that was written in the NUS CS Workshop of Real Time Rendering
//      hosted on Shadertoy https://www.shadertoy.com/view/wlsSzs
// * Analytical formulas by Inigo Quilez, source: http://iquilezles.org/www/articles/distfunctions/distfunctions.htm
// * glm source code on https://github.com/g-truc/glm

use crate::data::{Add, Length, Mat4, Minus, ScalarMul, Vec3, Vec4, _Mat};
use crate::shading::*;
use crate::shapes::{
    sdf_cube, sdf_cylinder, sdf_ellipsoid, sdf_plane, sdf_rounded_cylinder, sdf_sphere,
};
use crate::state::KeyboardMouseStates;
use crate::transformations::*;
use pixel_canvas::input::glutin::event::VirtualKeyCode;
use pixel_canvas::{Canvas, Color};
use rayon::prelude::*;
use std::io::stdin;
use std::process::exit;
use std::time::Instant;

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
    Panning,
    FreeMove,
    Zoom,
    Select,
}

fn main() {
    const VIEW_PLANE_WIDTH: f32 = 4.;
    const VIEW_PLANE_HEIGHT: f32 = 3.;
    const SELECT_CIRCLE_RADIUS_SQUARE: i32 = 5 * 5;
    let orthogonal_ray_dir_ec = Vec3::new_xyz(0., 0., 1.);
    let x_axis = Vec3::new_xyz(1., 0., 0.);
    let y_axis = Vec3::new_xyz(0., 1., 0.);
    let z_axis = Vec3::new_xyz(0., 0., 1.);
    let use_perspective;
    loop {
        println!("Use Perspective? y for perspective view, n for orthogonal view");
        let mut buf = String::new();
        let result = stdin().read_line(&mut buf);
        match result {
            Ok(_) => {
                if buf.len() <= 3 {
                    let s = buf.to_lowercase();
                    if s.contains("y") {
                        use_perspective = true;
                        println!("Using Perspective");
                        break;
                    } else if s.contains("n") {
                        use_perspective = false;
                        println!("Using Orthogonal");
                        break;
                    } else {
                        eprintln!("Wrong Input, try again");
                    }
                } else {
                    println!("Entered too many characters, try again");
                }
            }
            Err(e) => {
                eprintln!("Unexpected Error, exiting");
                eprintln!("{}", e);
                exit(-1);
            }
        }
    }

    let mut objects = Vec::new();
    let mut lights = Vec::new();
    let mut materials = Vec::new();
    init_scene(&mut objects, &mut materials, &mut lights);
    let fov_radian = (2.0_f32).atan() * 2.;

    let mut eye_pos = Vec3::new_xyz(0.0, 0.0, -1.0);
    let eye_pos_original = eye_pos.clone();
    let mut center = Vec3::new(0.);
    let center_original = center.clone();
    let mut up = Vec3::new_xyz(0.0, 1.0, 0.0);
    let up_original = up.clone();

    let dw = VIEW_PLANE_WIDTH / WIDTH_F;
    let dh = VIEW_PLANE_HEIGHT / HEIGHT_F;

    let now = Instant::now();
    // configure the window/canvas
    let canvas = Canvas::new(WIDTH, HEIGHT)
        .title("Dynamic Raytracer")
        .state(KeyboardMouseStates::new())
        .input(KeyboardMouseStates::handle_input);
    let mut before = now.elapsed().as_millis();
    let mut after = before;

    let mut mode = Mode::Orbit;

    let mut theta: f32 = 0.; // wc the angle between -z and +x
    let mut phi: f32 = std::f32::consts::FRAC_PI_2; // wc complement angle of the angle between the line and z-x plane

    let mut selected_obj_idx: i32 = -1;

    // render up to 60fps
    canvas.render(move |state, frame_buffer_image| {
        // switching modes
        let mut switch_mode = false;
        if state.received_keycode {
            match state.keycode {
                VirtualKeyCode::Key1 => {
                    mode = Mode::Orbit;
                    println!("Chose Mode: {:?}", mode);
                    switch_mode = true;
                }
                VirtualKeyCode::Key2 => {
                    mode = Mode::Panning;
                    println!("Chose Mode: {:?}", mode);
                    switch_mode = true;
                }
                VirtualKeyCode::Key3 => {
                    mode = Mode::FreeMove;
                    println!("Chose Mode: {:?}", mode);
                    switch_mode = true
                }
                VirtualKeyCode::Z => {
                    mode = Mode::Zoom;
                    println!("Chose Mode: {:?}", mode);
                    switch_mode = true
                }
                VirtualKeyCode::X => {
                    mode = Mode::Select;
                    println!("Chose Mode: {:?}", mode);
                    switch_mode = true
                }
                VirtualKeyCode::C => {
                    selected_obj_idx = -1;
                    switch_mode = true;
                }
                _ => {}
            }
        }
        // Orbiting camera
        if !switch_mode && state.received_keycode && mode == Mode::Orbit {
            println!("Key Pressed: {:?}", state.keycode);
            match state.keycode {
                // for orbiting around the origin
                VirtualKeyCode::A => theta += (5.0_f32).to_radians(),
                VirtualKeyCode::D => theta -= (5.0_f32).to_radians(),
                VirtualKeyCode::W => {
                    phi -= (2.5_f32).to_radians();
                    if phi <= 0. {
                        phi = 0.;
                    }
                }
                VirtualKeyCode::S => {
                    phi += (2.5_f32).to_radians();
                    if phi >= std::f32::consts::PI {
                        phi = std::f32::consts::PI;
                    }
                }
                VirtualKeyCode::R => {
                    theta = 0.;
                    phi = std::f32::consts::FRAC_PI_2;
                    center = center_original.clone();
                }
                _ => {}
            }
            let radius = eye_pos._minus(&center).get_length();
            eye_pos.set_y(phi.cos() * radius);
            eye_pos.set_z(-theta.cos() * radius * phi.sin());
            eye_pos.set_x(theta.sin() * radius * phi.sin());
        }
        // Moving camera in wc
        if !switch_mode && state.received_keycode && mode == Mode::FreeMove {
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
        before = now.elapsed().as_millis();
        let look_at_mat = look_at(&eye_pos, &center, &up);
        // bottom-left(0,0) top-right(w, h)
        let cursor_position = (state.x as i32, state.y as i32);
        // selecting object
        if state.received_mouse_press {
            println!(
                "Mouse Pressed at ({}, {})",
                cursor_position.0, cursor_position.1
            );
            let frag_coord = [cursor_position.0 as f32, cursor_position.1 as f32];
            let ray: Ray = if use_perspective {
                get_ray_perspective(fov_radian, &look_at_mat, &eye_pos, &frag_coord)
            } else {
                get_ray_orthogonal(
                    dw,
                    dh,
                    &orthogonal_ray_dir_ec,
                    &eye_pos,
                    &look_at_mat,
                    &frag_coord,
                )
            };
            let hit_object_idx = cast_hit_ray(&ray, &objects);
            match hit_object_idx {
                Some((idx, hit_pos)) => {
                    let obj: &Object = objects.get(idx as usize).unwrap();
                    match mode {
                        Mode::Zoom => {
                            println!(
                                "Focused on hit position ({}, {}, {})",
                                hit_pos.x(),
                                hit_pos.y(),
                                hit_pos.z()
                            );
                            center = hit_pos.clone();
                            // may need to adjust parameter according to camera move mode?
                            // unimplemented!()
                        }
                        Mode::Select => {
                            println!("Selected Object(idx={}, type={:?})", idx, obj.shape,);
                            selected_obj_idx = idx;
                        }
                        _ => println!(
                            "Mouse Pressed at ({}, {})",
                            cursor_position.0, cursor_position.1
                        ),
                    }
                }
                None => {}
            }
        }

        let mut rotating_or_scaling = false;
        if !switch_mode && state.received_keycode && selected_obj_idx != -1 {
            let obj = objects.get_mut(selected_obj_idx as usize).unwrap();
            let identity = Mat4::identity();
            match state.keycode {
                // Rotate around x
                VirtualKeyCode::I => {
                    let rotate = rotate_obj(identity, (5.0_f32).to_radians(), x_axis.clone());
                    let new_transformation = rotate.dot_mat(&obj.transformation);
                    obj.transformation = new_transformation;
                    rotating_or_scaling = true;
                }
                VirtualKeyCode::K => {
                    let rotate = rotate_obj(identity, (5.0_f32).to_radians(), x_axis.clone());
                    let new_transformation = rotate.dot_mat(&obj.transformation);
                    obj.transformation = new_transformation;
                    rotating_or_scaling = true;
                }
                // Rotate around y
                VirtualKeyCode::J => {
                    let rotate = rotate_obj(identity, (5.0_f32).to_radians(), y_axis.clone());
                    let new_transformation = rotate.dot_mat(&obj.transformation);
                    obj.transformation = new_transformation;
                    rotating_or_scaling = true;
                }
                VirtualKeyCode::L => {
                    let rotate = rotate_obj(identity, (5.0_f32).to_radians(), y_axis.clone());
                    let new_transformation = rotate.dot_mat(&obj.transformation);
                    obj.transformation = new_transformation;
                    rotating_or_scaling = true;
                }
                // Rotate around z
                VirtualKeyCode::U => {
                    let rotate = rotate_obj(identity, (5.0_f32).to_radians(), z_axis.clone());
                    let new_transformation = rotate.dot_mat(&obj.transformation);
                    obj.transformation = new_transformation;
                    rotating_or_scaling = true;
                }
                VirtualKeyCode::O => {
                    let rotate = rotate_obj(identity, (-5.0_f32).to_radians(), z_axis.clone());
                    let new_transformation = rotate.dot_mat(&obj.transformation);
                    obj.transformation = new_transformation;
                    rotating_or_scaling = true;
                }
                // Scale
                VirtualKeyCode::PageUp => {
                    obj.transformation = scale(&obj.transformation, 1.1);
                    rotating_or_scaling = true;
                }
                VirtualKeyCode::PageDown => {
                    obj.transformation = scale(&obj.transformation, 0.85);
                    rotating_or_scaling = true;
                }
                // reset
                VirtualKeyCode::M => {
                    obj.transformation = obj.original_transformation.clone();
                    rotating_or_scaling = true;
                }
                _ => {}
            }
        }
        // zooming
        if !rotating_or_scaling && !switch_mode && state.received_keycode && mode == Mode::Zoom {
            let mut camera_focus_direction = look_at_mat._get_column(2);
            match state.keycode {
                VirtualKeyCode::Q => {
                    camera_focus_direction.scalar_mul_(0.1);
                    eye_pos.add_(&camera_focus_direction);
                }
                VirtualKeyCode::E => {
                    camera_focus_direction.scalar_mul_(0.1);
                    eye_pos.minus_(&camera_focus_direction);
                }
                _ => {}
            }
        }
        // panning camera or the selected object
        if !rotating_or_scaling && !switch_mode && state.received_keycode && mode == Mode::Panning {
            let mut camera_up = look_at_mat._get_column(1);
            let mut camera_right = look_at_mat._get_column(0);
            if selected_obj_idx != -1 {
                let obj = objects.get_mut(selected_obj_idx as usize).unwrap();
                match state.keycode {
                    // for panning
                    VirtualKeyCode::W => {
                        camera_up.scalar_mul_(0.1);
                        let new_transformation =
                            translate_obj(obj.transformation.clone(), &camera_up);
                        obj.transformation = new_transformation;
                    }
                    VirtualKeyCode::S => {
                        camera_up.scalar_mul_(-0.1);
                        let new_transformation =
                            translate_obj(obj.transformation.clone(), &camera_up);
                        obj.transformation = new_transformation;
                    }
                    VirtualKeyCode::A => {
                        camera_right.scalar_mul_(-0.1);
                        let new_transformation =
                            translate_obj(obj.transformation.clone(), &camera_right);
                        obj.transformation = new_transformation;
                    }
                    VirtualKeyCode::D => {
                        camera_right.scalar_mul_(0.1);
                        let new_transformation =
                            translate_obj(obj.transformation.clone(), &camera_right);
                        obj.transformation = new_transformation;
                    }
                    VirtualKeyCode::R => {
                        obj.transformation = obj.original_transformation.clone();
                    }
                    _ => {}
                }
            } else {
                match state.keycode {
                    // for panning
                    VirtualKeyCode::W => {
                        camera_up.scalar_mul_(0.1);
                        eye_pos.add_(&camera_up);
                        center.add_(&camera_up);
                    }
                    VirtualKeyCode::S => {
                        camera_up.scalar_mul_(0.1);
                        eye_pos.minus_(&camera_up);
                        center.minus_(&camera_up);
                    }
                    VirtualKeyCode::A => {
                        camera_right.scalar_mul_(0.1);
                        eye_pos.minus_(&camera_right);
                        center.minus_(&camera_right);
                    }
                    VirtualKeyCode::D => {
                        camera_right.scalar_mul_(0.1);
                        eye_pos.add_(&camera_right);
                        center.add_(&camera_right);
                    }
                    VirtualKeyCode::R => {
                        eye_pos = eye_pos_original.clone();
                        center = center_original.clone();
                    }
                    _ => {}
                }
            }
        }

        frame_buffer_image
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, pixel)| {
                let y = idx / WIDTH;
                let x = idx % WIDTH;
                let dx = x as i32 - cursor_position.0;
                let dy = y as i32 - cursor_position.1;
                let dist = dx * dx + dy * dy;
                let in_circle = dist < SELECT_CIRCLE_RADIUS_SQUARE;
                if in_circle {
                    *pixel = Color {
                        r: 255,
                        g: 255,
                        b: 255,
                    };
                    return;
                }
                let frag_coord = [x as f32, y as f32];
                let primary_ray: Ray = if use_perspective {
                    get_ray_perspective(fov_radian, &look_at_mat, &eye_pos, &frag_coord)
                } else {
                    get_ray_orthogonal(
                        dw,
                        dh,
                        &orthogonal_ray_dir_ec,
                        &eye_pos,
                        &look_at_mat,
                        &frag_coord,
                    )
                };
                *pixel = to_color(shade(primary_ray, &objects, &materials, &lights));
            });
        after = now.elapsed().as_millis();
        println!("Took {} ms to render one frame", after - before);
        state.reset_flags();
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
