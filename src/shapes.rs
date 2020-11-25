use crate::data::{Vec3, Length, VecDot, Minus};

pub fn sdf_sphere(ref_pos: &Vec3, radius: f32) -> f32 {
    ref_pos.get_length() - radius
}

pub fn sdf_plane(ref_pos: &Vec3, normal: &Vec3, w: f32) -> f32
{
    return ref_pos.dot(normal) + w;
}

pub fn sdf_cube(ref_pos: &Vec3, dimensions: Vec3) -> f32
{
    let v = Vec3::new_xyz(ref_pos.x().abs(), ref_pos.y().abs(), ref_pos.z().abs());
    let d = v._minus(&dimensions);
    let d_max = Vec3::new_xyz(d.x().max(0.0), d.y().max(0.0), d.z().max(0.0));
    let dist = d_max.get_length() + f32::min(f32::max(d.x(), f32::max(d.y(), d.z())), 0.0);
    return dist;
}

pub fn sdf_ellipsoid(ref_pos: &Vec3, dimensions: &Vec3) -> f32
{
    let k0v = Vec3::new_xyz(ref_pos.x() / dimensions.x(), ref_pos.y() / dimensions.y(), ref_pos.z() / dimensions.z());
    let k0 = k0v.get_length();
    let k1v = Vec3::new_xyz(ref_pos.x() / (dimensions.x() * dimensions.x()),
                            ref_pos.y() / (dimensions.y() * dimensions.y()),
                            ref_pos.z() / (dimensions.z() * dimensions.z()));
    let k1 = k1v.get_length();
    return k0 * (k0 - 1.) / k1;
}

pub fn sdf_rounded_cylinder(ref_pos: &Vec3, radius: f32, round_radius: f32, height: f32) -> f32
{
    let d1 = (ref_pos.x() * ref_pos.x() + ref_pos.z() * ref_pos.z()).sqrt() - 2. * radius + round_radius;
    let d2 = ref_pos.y().abs() - height;
    let d1_max = d1.max(0.);
    let d2_max = d2.max(0.);
    return f32::min(f32::max(d1, d2), 0.) + (d1_max * d1_max + d2_max * d2_max).sqrt() - round_radius;
}

pub fn sdf_cylinder(ref_pos: &Vec3, radius: f32, height: f32) -> f32
{
    let dx = (ref_pos.x() * ref_pos.x() + ref_pos.z() * ref_pos.z()).sqrt() - height;
    let dy = ref_pos.y().abs() - radius;
    let dx_max = dx.max(0.0);
    let dy_max = dy.max(0.0);
    return f32::min(f32::max(dx, dy), 0.0) + (dx_max * dx_max + dy_max * dy_max).sqrt();
}

