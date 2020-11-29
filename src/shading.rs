use rand::prelude::*;

use crate::data::{
    Add, Cross, Mat3, Mat4, MatVecDot, Minus, Normalize, Product, ScalarMul, Vec3, Vec4, VecDot,
};
use crate::*;

pub struct Scene {
    pub objects: Vec<Object>,
    pub lights: Vec<Light>,
    pub materials: Vec<Material>,
}

#[derive(Debug, Copy, Clone)]
pub struct Pixel {
    pub x: usize,
    pub y: usize,
    pub x_f: f32,
    pub y_f: f32,
    pub alpha: f32,
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
            alpha,
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

    pub fn set_alpha(&mut self, alpha: f32) {
        self.alpha = alpha;
        self.ema_r.set_alpha(alpha);
        self.ema_g.set_alpha(alpha);
        self.ema_b.set_alpha(alpha);
    }
}

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
    pub transformation: Mat4,
    pub shape: ShapeTypes,
    pub material_id: usize,
    pub original_transformation: Mat4,
}

#[derive(Copy, Clone)]
pub struct Material {
    pub ambient: Vec3,
    pub diffuse: Vec3,
    pub reflection: Vec3,
    pub global_reflection: Vec3,
    pub specular: f32,
}

#[derive(Copy, Clone)]
pub struct Light {
    pub position: Vec3,
    pub original_position: Vec3,
    pub ambient: Vec3,
    pub diffuse: Vec3,
    pub r: f32,
}

fn get_jittered_pos(pos: &Vec3, right: &Vec3, up: &Vec3, width: f32, height: f32) -> Vec3 {
    let mut random = rand::thread_rng();
    let mut x = random.gen::<f32>() * 2.0 - 1.0;
    let mut y = random.gen::<f32>() * 2.0 - 1.0;
    x *= width / 2.0;
    y *= height / 2.0;
    let r = right.scalar_mul(x);
    let u = up.scalar_mul(y);
    let mut p = pos._add(&r);
    p.add_(&u);
    return p;
}

fn get_jittered_light_pos(light: &Light, pos: &Vec3) -> Vec3 {
    let light_basis = look_at(&light.position, pos, &Vec3::new_xyz(0., 0.0, 1.0));
    let mut random = rand::thread_rng();
    let mut x = random.gen::<f32>() * 2.0 - 1.0;
    let mut y = (1.0 - x * x).sqrt();
    let r2 = random.gen::<f32>() * 2.0 - 1.0;
    y *= r2;
    y *= light.r;
    x *= light.r;
    let mut camera_up = light_basis._get_column(1);
    let mut camera_right = light_basis._get_column(0);
    camera_up.scalar_mul_(y);
    camera_right.scalar_mul_(x);
    let mut light_pos = light.position.clone();
    light_pos.add_(&camera_up);
    light_pos.add_(&camera_right);
    return light_pos;
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

fn _look_at(look_at_direction: &Vec3, up: &Vec3) -> Mat3 {
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

pub fn look_at(eye: &Vec3, center: &Vec3, up: &Vec3) -> Mat3 {
    let mut look_at_direction = center._minus(eye);
    look_at_direction.normalize_();
    return _look_at(&look_at_direction, up);
}

#[inline]
pub fn ray_direction_perspective(fovy_radian: f32, frag_coord: &[f32; 2]) -> Vec3 {
    // ignore the case when the aspect of view != aspect of window
    let x = frag_coord[0] - WIDTH_HF + 0.5;
    let y = frag_coord[1] - HEIGHT_HF + 0.5;
    let z = HEIGHT_F / (fovy_radian / 2.).tan();
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

const SPHERE_RADIUS: usize = 100;
const SPHERE_RADIUS_F: f32 = SPHERE_RADIUS as f32;

pub fn cast_ray(
    ray: &Ray,
    pre_obj: i32,
    scene: &Scene,
    has_hit: &mut bool,
    hit_pos: &mut Vec3,
    hit_normal: &mut Vec3,
    k_rg: &mut Vec3,
    hit_obj: &mut i32,
    enable_soft_shadow: bool,
    enable_env_mapping: bool,
    env_tex: &Tex2D,
) -> Vec3 {
    let objects = &scene.objects;
    let (obj_idx, dist) = shortest_dist_to_surface(objects, &ray.origin, &ray.direction, pre_obj);
    if dist > MAX_DIST - EPSILON {
        *has_hit = false;
        if enable_env_mapping {
            let mut pos = ray.origin.clone();
            let dir = ray.direction.clone();
            let mut dist = SPHERE_RADIUS_F - pos.get_length();
            let mut step = 0;
            while dist > EPSILON && step < MAX_MARCHING_STEPS {
                pos.add_(&dir.scalar_mul(dist));
                dist = SPHERE_RADIUS_F - pos.get_length();
                step += 1;
            }
            let theta = (pos.z() / SPHERE_RADIUS_F).acos();
            let mut phi = (pos.y() / pos.x()).atan() * 2.0 + std::f32::consts::PI;
            if pos.x() == 0.0 {
                phi = if pos.y() > 0.0 {
                    std::f32::consts::FRAC_PI_4
                } else {
                    std::f32::consts::FRAC_PI_2 + std::f32::consts::FRAC_PI_4
                };
            }
            let v = theta / std::f32::consts::PI;
            let u = phi / (2.0 * std::f32::consts::PI);
            let color = env_tex.get_color_f(u, v);
            return color;
        } else {
            return Vec3::new_rgb(BACKGROUND_COLOR.0, BACKGROUND_COLOR.1, BACKGROUND_COLOR.2);
        }
    } else {
        *hit_obj = obj_idx;
        let obj = objects.get(obj_idx as usize).unwrap();
        *has_hit = true;
        let ref_pos = ray.origin._add(&ray.direction.scalar_mul(dist));
        *hit_pos = ref_pos.clone();
        *hit_normal = estimate_normal(&ref_pos, objects);
        let hit_material = scene.materials.get(obj.material_id).unwrap();
        *k_rg = hit_material.global_reflection.clone();
        let mut local_color = Vec3::new(0.);
        for l in scene.lights.iter() {
            let shadow_ray = if enable_soft_shadow {
                let light_pos = get_jittered_light_pos(l, hit_pos);
                light_pos._minus(hit_pos)
            } else {
                l.position._minus(hit_pos)
            };
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
    let view_init_direction = ray_direction_perspective(fov_radian, &frag_coord); // in eye space
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

const GLOSSY_LIGHT_NUM: usize = 3;
const JITTER_DIM: f32 = 0.001;

pub fn shade(
    primary_ray: Ray,
    scene: &Scene,
    enable_soft_shadow: bool,
    enable_glossy: bool,
    enable_env_mapping: bool,
    env_tex: &Tex2D,
) -> Vec3 {
    let eps = Vec3::new(EPSILON);
    let mut next_ray = primary_ray.clone();
    let mut color_result = Vec3::new(0.);
    let mut component_k_rg = Vec3::new(1.);
    let mut pre_obj = -1;
    let UP = Vec3::new_xyz(0.0, 0.0, 1.0);
    for level in 0..NUM_ITERATIONS {
        let mut has_hit = false;
        let mut hit_pos = Vec3::_new();
        let mut hit_normal = Vec3::_new();
        let mut k_rg = Vec3::new(1.);
        let color_local = if enable_glossy && level != 0 {
            let ray_basis = _look_at(&next_ray.direction, &UP);
            let ray_up = ray_basis._get_column(1);
            let ray_right = ray_basis._get_column(0);
            let mut base_point = next_ray.direction.scalar_mul(1.0);
            base_point.add_(&next_ray.origin);
            let glossy_colors: Vec<Vec3> = (0..GLOSSY_LIGHT_NUM)
                .into_iter()
                .map(|_| {
                    let jitter_point =
                        get_jittered_pos(&base_point, &ray_right, &ray_up, JITTER_DIM, JITTER_DIM);
                    let mut dir = jitter_point._minus(&next_ray.direction);
                    dir.normalize_();
                    let glossy_ray = Ray {
                        direction: dir,
                        origin: next_ray.origin.clone(),
                    };
                    let mut temp1 = Vec3::new(0.0);
                    let mut temp2 = temp1.clone();
                    let mut temp3 = temp1.clone();
                    let color = cast_ray(
                        &glossy_ray,
                        pre_obj,
                        scene,
                        &mut false,
                        &mut temp1,
                        &mut temp2,
                        &mut temp3,
                        &mut 0,
                        false,
                        enable_env_mapping,
                        env_tex,
                    );
                    return color;
                })
                .collect();
            let mut color = cast_ray(
                &next_ray,
                pre_obj,
                scene,
                &mut has_hit,
                &mut hit_pos,
                &mut hit_normal,
                &mut k_rg,
                &mut pre_obj,
                enable_soft_shadow,
                enable_env_mapping,
                env_tex,
            );
            glossy_colors.iter().for_each(|c| color.add_(c));
            color.scalar_div_((GLOSSY_LIGHT_NUM + 1) as f32);
            color
        } else {
            cast_ray(
                &next_ray,
                pre_obj,
                scene,
                &mut has_hit,
                &mut hit_pos,
                &mut hit_normal,
                &mut k_rg,
                &mut pre_obj,
                enable_soft_shadow,
                enable_env_mapping,
                env_tex,
            )
        };
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
