use crate::data::{Vec3, Length, Vec4, Normalize, VecDot, Minus};

pub trait Shape
{
    fn get_dist(&self, ref_pos: &Vec3) -> f32;
}

pub struct Sphere
{
    radius: f32
}

impl Sphere
{
    pub fn new(radius: f32) -> Box<Self>
    {
        Box::new(Sphere {
            radius
        })
    }
}

impl Shape for Sphere
{
    fn get_dist(&self, ref_pos: &Vec3) -> f32 {
        ref_pos.get_length() - self.radius
    }
}


pub struct Plane
{
    coefficients: Vec4,
    c3: Vec3
}

impl Plane
{
    pub fn new(a: f32, b: f32, c: f32, d: f32) -> Box<Self>
    {
        let mut dimensions = Vec4::new_xyzw(a, b, c, d);
        dimensions.normalize_();
        Box::new(Plane {
            coefficients: dimensions,
            c3: Vec3::from(&dimensions)
        })
    }
}

impl Shape for Plane
{
    fn get_dist(&self, ref_pos: &Vec3) -> f32 {
        return ref_pos.dot(&self.c3) + self.coefficients.w();
    }
}

pub struct Cube
{
    dimensions: Vec3
}

impl Cube
{
    pub fn new(width: f32, height: f32, depth: f32) -> Box<Self>
    {
        Box::new(Cube {
            dimensions: Vec3::new_xyz(width, height, depth)
        })
    }
}

impl Shape for Cube
{
    fn get_dist(&self, ref_pos: &Vec3) -> f32 {
        let v = Vec3::new_xyz(ref_pos.x().abs(), ref_pos.y().abs(), ref_pos.z().abs());
        let d = v._minus(&self.dimensions);
        let d_max = Vec3::new_xyz(d.x().max(0.0), d.y().max(0.0), d.z().max(0.0));
        let dist = d_max.get_length() + f32::min(f32::max(d.x(), f32::max(d.y(), d.z())), 0.0);
        return dist;
    }
}

pub struct Ellipsoid
{
    dimensions: Vec3
}

impl Ellipsoid
{
    pub fn new(dimensions: Vec3) -> Box<Self>
    {
        Box::new(Ellipsoid {
            dimensions
        })
    }
}

impl Shape for Ellipsoid
{
    fn get_dist(&self, ref_pos: &Vec3) -> f32 {
        let k0v = Vec3::new_xyz(ref_pos.x() / self.dimensions.x(), ref_pos.y() / self.dimensions.y(), ref_pos.z() / self.dimensions.z());
        let k0 = k0v.get_length();
        let k1v = Vec3::new_xyz(ref_pos.x() / (self.dimensions.x() * self.dimensions.x()),
                                ref_pos.y() / (self.dimensions.y() * self.dimensions.y()),
                                ref_pos.z() / (self.dimensions.z() * self.dimensions.z()));
        let k1 = k1v.get_length();
        return k0 * (k0 - 1.) / k1;
    }
}