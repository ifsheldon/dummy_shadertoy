// Code Reference:
// * My fragment shader code that was written in the NUS CS Workshop of Real Time Rendering
//      hosted on Shadertoy https://www.shadertoy.com/view/wlsSzs
// * Analytical formulas by Inigo Quilez, source: http://iquilezles.org/www/articles/distfunctions/distfunctions.htm
// * glm source code on https://github.com/g-truc/glm

use crate::data::{
    Add, Cross, Mat3, Mat4, MatVecDot, Minus, Normalize, Product, ScalarMul, Vec3, Vec4, VecDot,
};
use crate::shapes::{
    sdf_cube, sdf_cylinder, sdf_ellipsoid, sdf_plane, sdf_rounded_cylinder, sdf_sphere,
};
use crate::state::KeyboardMouseStates;
use pixel_canvas::input::glutin::event::VirtualKeyCode;
use pixel_canvas::{Canvas, Color};
use rayon::prelude::*;
use std::io::stdin;
use std::process::exit;
use std::time::Instant;

mod data;
mod err;
mod shapes;
mod state;

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

#[derive(Copy, Clone)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

#[derive(Debug)]
pub enum ShapeTypes {
    Sphere(f32),
    RoundedCylinder(f32, f32, f32),
    Plane(f32, f32, f32, f32),
    Cylinder(f32, f32),
    Cube(f32, f32, f32),
    Ellipsoid(f32, f32, f32),
}

pub struct Object {
    transformation: Mat4,
    shape: ShapeTypes,
    material_id: usize,
}

#[derive(Copy, Clone)]
pub struct Material {
    ambient: Vec3,
    diffuse: Vec3,
    reflection: Vec3,
    global_reflection: Vec3,
    specular: f32,
}

#[derive(Copy, Clone)]
pub struct Light {
    position: Vec3,
    ambient: Vec3,
    diffuse: Vec3,
}

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

pub fn translate_obj(mat: Mat4, translation: &Vec3) -> Mat4 {
    let mut result = mat.clone();
    let translation = translation.scalar_mul(-1.);
    let m0 = mat._get_column(0);
    let m1 = mat._get_column(1);
    let m2 = mat._get_column(2);
    let m3 = mat._get_column(3);
    let mut m0t0 = m0.scalar_mul(translation.x());
    let m1t1 = m1.scalar_mul(translation.y());
    let m2t2 = m2.scalar_mul(translation.z());
    m0t0.add_(&m1t1);
    m0t0.add_(&m2t2);
    m0t0.add_(&m3);
    result._set_column(3, &m0t0);
    return result;
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

#[inline]
pub fn union(distances: Vec<f32>) -> f32 {
    unsafe {
        let mut min_dist = *distances.get_unchecked(0);
        for i in 0..distances.len() {
            min_dist = f32::min(min_dist, *distances.get_unchecked(i));
        }
        return min_dist;
    }
}

pub fn union_obj(distances: &Vec<f32>) -> (i32, f32) {
    unsafe {
        let mut min_dist = *distances.get_unchecked(0);
        let mut idx = 0;
        for i in 0..distances.len() {
            let c = *distances.get_unchecked(i);
            if c < min_dist {
                min_dist = c;
                idx = i;
            }
        }
        return (idx as i32, min_dist);
    }
}

pub fn calc_dist(ref_pos: &Vec3, obj: &Object) -> f32 {
    let ref_p = Vec4::from(ref_pos, 1.);
    let mut ref_p = obj.transformation.mat_vec_dot(&ref_p);
    ref_p.scalar_mul_(1. / ref_p.w());
    let ref_point = &Vec3::from(&ref_p);
    return match obj.shape {
        ShapeTypes::Sphere(r) => sdf_sphere(ref_point, r),
        ShapeTypes::RoundedCylinder(r, rr, h) => sdf_rounded_cylinder(ref_point, r, rr, h),
        ShapeTypes::Plane(a, b, c, w) => sdf_plane(ref_point, &Vec3::new_xyz(a, b, c), w),
        ShapeTypes::Cylinder(r, h) => sdf_cylinder(ref_point, r, h),
        ShapeTypes::Cube(w, h, d) => sdf_cube(ref_point, Vec3::new_xyz(w, h, d)),
        ShapeTypes::Ellipsoid(a, b, c) => sdf_ellipsoid(ref_point, &Vec3::new_xyz(a, b, c)),
    };
}

#[inline]
pub fn scene_distances(ref_pos: &Vec3, objects: &Vec<Object>) -> Vec<f32> {
    let dis = objects.iter().map(|o| calc_dist(ref_pos, o)).collect();
    return dis;
}

#[inline]
pub fn scene_sdf(ref_pos: &Vec3, objects: &Vec<Object>) -> f32 {
    union(scene_distances(ref_pos, objects))
}

pub fn shortest_dist_to_surface(
    objects: &Vec<Object>,
    eye: &Vec3,
    direction: &Vec3,
    pre_obj: i32,
) -> (i32, f32) {
    let mut depth = MIN_DIST;
    for _ in 0..MAX_MARCHING_STEPS {
        let mut distances = scene_distances(&eye._add(&direction.scalar_mul(depth)), objects);
        if pre_obj != -1 {
            *distances.get_mut(pre_obj as usize).unwrap() = 2. * MAX_DIST;
        }
        let (hit_obj_idx, dist) = union_obj(&distances);
        if dist < EPSILON {
            return (hit_obj_idx, depth);
        }
        depth += dist;
        if depth >= MAX_DIST {
            return (objects.len() as i32, MAX_DIST);
        }
    }
    return (objects.len() as i32, MAX_DIST);
}

pub fn look_at(eye: &Vec3, center: &Vec3, up: &Vec3) -> Mat3 {
    let mut look_at_direction = center._minus(eye);
    look_at_direction.normalize_();
    let mut right = look_at_direction.cross(up);
    right.normalize_();
    let mut camera_up = right.cross(&look_at_direction);
    camera_up.normalize_();
    let mut m = Mat3::identity();
    let f = look_at_direction;

    m._set_column(0, &right);
    m._set_column(1, &camera_up);
    m._set_column(2, &f);

    return m;
}

#[inline]
pub fn ray_direction_perspective(_fov_radian: f32, frag_coord: &[f32; 2]) -> Vec3 {
    let x = frag_coord[0] - WIDTH_HF + 0.5;
    let y = frag_coord[1] - HEIGHT_HF + 0.5;
    // let z = HEIGHT_F / (fov_radian / 2.).tan();
    let z = HEIGHT_F / 2.; // 2.0 = (VIEW_PLANE_WIDTH /2) / |-1| (cam position)
    let mut v = Vec3::new_xyz(x, y, z);
    v.normalize_();
    return v;
}

#[inline]
pub fn estimate_normal(p: &Vec3, objects: &Vec<Object>) -> Vec3 {
    let mut poses = Vec::new();
    poses.push((0, Vec3::new_xyz(p.x() + EPSILON, p.y(), p.z())));
    poses.push((1, Vec3::new_xyz(p.x() - EPSILON, p.y(), p.z())));

    poses.push((2, Vec3::new_xyz(p.x(), p.y() + EPSILON, p.z())));
    poses.push((3, Vec3::new_xyz(p.x(), p.y() - EPSILON, p.z())));

    poses.push((4, Vec3::new_xyz(p.x(), p.y(), p.z() + EPSILON)));
    poses.push((5, Vec3::new_xyz(p.x(), p.y(), p.z() - EPSILON)));

    let mut sdfs: Vec<(i32, f32)> = poses
        .iter()
        .map(|(idx, pos)| (idx.clone(), scene_sdf(pos, objects)))
        .collect(); // seems to take more time when using par_iter, but worth trying when porting to higher performance computer

    sdfs.sort_by(|x, y| i32::cmp(&x.0, &y.0));

    let mut v = Vec3::new_xyz(
        sdfs.get(0).unwrap().1 - sdfs.get(1).unwrap().1,
        sdfs.get(2).unwrap().1 - sdfs.get(3).unwrap().1,
        sdfs.get(4).unwrap().1 - sdfs.get(5).unwrap().1,
    );
    v.normalize_();
    return v;
}

pub fn reflect(incident_vec: &Vec3, normalized_normal: &Vec3) -> Vec3 {
    incident_vec._minus(&normalized_normal.scalar_mul(2.0 * normalized_normal.dot(incident_vec)))
}

pub fn phong_lighting(
    light_direction: &Vec3,
    normalized_normal: &Vec3,
    view_direction: &Vec3,
    in_shadow: bool,
    material: &Material,
    light: &Light,
) -> Vec3 {
    if in_shadow {
        return light.ambient.product(&material.ambient);
    } else {
        let reflected_light = reflect(&light_direction.scalar_mul(-1.), normalized_normal);
        let n_dot_l = f32::max(0.0, normalized_normal.dot(light_direction));
        let r_dot_l = f32::max(0.0, reflected_light.dot(view_direction));
        let r_dot_v_pow_n = if r_dot_l == 0.0 {
            0.0
        } else {
            r_dot_l.powf(material.specular)
        };
        let ambient = light.ambient.product(&material.ambient);
        let mut result = material.diffuse.scalar_mul(n_dot_l);
        result.add_(&material.reflection.scalar_mul(r_dot_v_pow_n));
        result.product_(&light.diffuse);
        result.add_(&ambient);
        return result;
    }
}

pub fn cast_hit_ray(ray: &Ray, objects: &Vec<Object>) -> Option<i32> {
    let (obj_idx, dist) = shortest_dist_to_surface(objects, &ray.origin, &ray.direction, -1);
    return if dist > MAX_DIST - EPSILON {
        None
    } else {
        Some(obj_idx)
    };
}

pub fn cast_ray(
    ray: &Ray,
    pre_obj: i32,
    lights: &Vec<Light>,
    materials: &Vec<Material>,
    objects: &Vec<Object>,
    has_hit: &mut bool,
    hit_pos: &mut Vec3,
    hit_normal: &mut Vec3,
    k_rg: &mut Vec3,
    hit_obj: &mut i32,
) -> Vec3 {
    let (obj_idx, dist) = shortest_dist_to_surface(objects, &ray.origin, &ray.direction, pre_obj);
    if dist > MAX_DIST - EPSILON {
        *has_hit = false;
        return Vec3::new_rgb(BACKGROUND_COLOR.0, BACKGROUND_COLOR.1, BACKGROUND_COLOR.2);
    } else {
        *hit_obj = obj_idx;
        let obj = objects.get(obj_idx as usize).unwrap();
        *has_hit = true;
        let ref_pos = ray.origin._add(&ray.direction.scalar_mul(dist));
        *hit_pos = ref_pos.clone();
        *hit_normal = estimate_normal(&ref_pos, objects);
        let hit_material = materials.get(obj.material_id).unwrap();
        *k_rg = hit_material.global_reflection.clone();
        let mut local_color = Vec3::new(0.);
        for l in lights {
            let shadow_ray = l.position._minus(hit_pos);
            let s_ray = Ray {
                origin: hit_pos.clone(),
                direction: shadow_ray.normalize(),
            };
            let (_hit_obj_index, d) =
                shortest_dist_to_surface(objects, &s_ray.origin, &s_ray.direction, obj_idx);
            let hit_sth = d < MAX_DIST - EPSILON;
            local_color.add_(&phong_lighting(
                &s_ray.direction,
                hit_normal,
                &ray.direction.scalar_mul(-1.),
                hit_sth,
                &hit_material,
                l,
            ));
        }
        return local_color;
    }
}

pub fn get_ray_perspective(
    fov_radian: f32,
    look_at_mat: &Mat3,
    eye_pos_wc: &Vec3,
    frag_coord: &[f32; 2],
) -> Ray {
    let view_init_direction = ray_direction_perspective(fov_radian, &frag_coord);
    let wc_ray_dir = look_at_mat.mat_vec_dot(&view_init_direction);
    let primary_ray = Ray {
        origin: eye_pos_wc.clone(),
        direction: wc_ray_dir.normalize(),
    };
    return primary_ray;
}

#[inline]
pub fn get_ray_orthogonal(
    dw: f32,
    dh: f32,
    _ray_dir_ec: &Vec3,
    eye_pos_wc: &Vec3,
    look_at: &Mat3,
    frag_coord: &[f32; 2],
) -> Ray {
    // let mut ray_dir_wc = look_at.mat_vec_dot(ray_dir_ec);
    let mut ray_dir_wc = look_at._get_column(2); // A little trick here
    ray_dir_wc.normalize_();
    let origin_ec = Vec3::new_xyz(
        dw * (frag_coord[0] - WIDTH_HF + 0.5),
        dh * (frag_coord[1] - HEIGHT_HF + 0.5),
        0.,
    );
    let mut origin_wc = look_at.mat_vec_dot(&origin_ec);
    origin_wc.add_(eye_pos_wc); // no need for multiplication for translations
    Ray {
        origin: origin_wc,
        direction: ray_dir_wc,
    }
}

pub fn shade(
    primary_ray: Ray,
    objects: &Vec<Object>,
    materials: &Vec<Material>,
    lights: &Vec<Light>,
) -> Vec3 {
    let eps = Vec3::new(EPSILON);
    let mut next_ray = primary_ray.clone();
    let mut color_result = Vec3::new(0.);
    let mut component_k_rg = Vec3::new(1.);
    let mut pre_obj = -1;
    for _level in 0..NUM_ITERATIONS {
        let mut has_hit = false;
        let mut hit_pos = Vec3::_new();
        let mut hit_normal = Vec3::_new();
        let mut k_rg = Vec3::new(1.);
        let color_local = cast_ray(
            &next_ray,
            pre_obj,
            lights,
            materials,
            objects,
            &mut has_hit,
            &mut hit_pos,
            &mut hit_normal,
            &mut k_rg,
            &mut pre_obj,
        );
        color_result.add_(&component_k_rg.product(&color_local));
        if !has_hit {
            break;
        }
        component_k_rg = component_k_rg.product(&k_rg);
        let mut next_ray_direction = reflect(&next_ray.direction, &hit_normal);
        next_ray_direction.normalize_();
        next_ray = Ray {
            origin: hit_pos._add(&eps),
            direction: next_ray_direction,
        };
    }
    return color_result;
}

fn main() {
    const VIEW_PLANE_WIDTH: f32 = 4.;
    const VIEW_PLANE_HEIGHT: f32 = 3.;
    const SELECT_CIRCLE_RADIUS_SQUARE: i32 = 5 * 5;
    let orthogonal_ray_dir_ec: Vec3 = Vec3::new_xyz(0., 0., 1.);
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
    let center = Vec3::new(0.);
    let up = Vec3::new_xyz(0.0, 1.0, 0.0);

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
    let mut theta: f32 = 0.;
    // render up to 60fps
    canvas.render(move |state, frame_buffer_image| {
        // bottom-left(0,0) top-right(w, h)
        if state.received_keycode {
            println!("Key Pressed: {:?}", state.keycode);
            match state.keycode {
                VirtualKeyCode::A => theta += (5.0_f32).to_radians(),
                VirtualKeyCode::D => theta -= (5.0_f32).to_radians(),
                VirtualKeyCode::R => theta = 0.,
                _ => {}
            }
        }
        // println!("Theta: {}", theta.to_degrees());
        before = now.elapsed().as_millis();
        // theta = (before as f32) / 1000.;
        eye_pos.set_z(-theta.cos());
        eye_pos.set_x(theta.sin());
        let look_at_mat = look_at(&eye_pos, &center, &up);
        let cursor_position = (state.x as i32, state.y as i32);
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
                Some(idx) => {
                    let obj: &Object = objects.get(idx as usize).unwrap();
                    println!("Selected Object(idx={}, type={:?})", idx, obj.shape);
                }
                None => {}
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

// fn main()
// {
//     let canvas = Canvas::new(512, 512)
//         .title("Tile")
//         .state(KeyboardMouseStates::new())
//         .input(KeyboardMouseStates::handle_input);
//     // The canvas will render for you at up to 60fps.
//     canvas.render(|state, image| {
//         // Modify the `image` based on your state.
//         let width = image.width() as usize;
//         let cursor_position = [state.x as i32, state.y as i32];
//         if state.received_mouse_press
//         {
//             println!("Mouse Pressed, ({}, {})", cursor_position[0], cursor_position[1]);
//         }
//         if state.received_keycode
//         {
//             println!("Key Got")
//         }
//         state.reset_flags();
//         for (y, row) in image.chunks_mut(width).enumerate() {
//             for (x, pixel) in row.iter_mut().enumerate() {
//                 let dx = x as i32 - cursor_position[0];
//                 let dy = y as i32 - cursor_position[1];
//                 let dist = dx * dx + dy * dy;
//                 *pixel = Color {
//                     r: if dist < 12 * 12 { dy as u8 } else { 0 },
//                     g: if dist < 12 * 12 { dx as u8 } else { 0 },
//                     b: (x * y) as u8,
//                 }
//             }
//         }
//     });
// }

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
