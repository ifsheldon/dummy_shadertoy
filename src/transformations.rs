use crate::data::{Add, Mat4, Normalize, ScalarMul, Vec3, Vec4, _Mat};

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

pub fn rotate_obj(transformation: Mat4, angle: f32, mut axis: Vec3) -> Mat4 {
    let angle = -angle;
    let cos = angle.cos();
    let sin = angle.sin();
    axis.normalize_();
    let temp = axis.scalar_mul(1. - cos);
    let mut rotate_mat = Mat4::identity();
    let mut column = Vec4::new(0.);
    // first column
    column.set_x(cos + temp.x() + axis.x());
    column.set_y(temp.x() * axis.y() + sin * axis.z());
    column.set_z(temp.x() * axis.z() - sin * axis.y());
    column.set_w(0.);
    rotate_mat._set_column(0, &column);
    // second column
    column.set_x(temp.y() * axis.x() - sin * axis.z());
    column.set_y(cos + temp.y() * axis.y());
    column.set_z(temp.y() * axis.z() + sin * axis.x());
    column.set_w(0.);
    rotate_mat._set_column(1, &column);
    // third column
    column.set_x(temp.z() * axis.x() + sin * axis.y());
    column.set_y(temp.z() * axis.y() - sin * axis.x());
    column.set_z(cos + temp.z() * axis.z());
    column.set_w(0.);
    rotate_mat._set_column(2, &column);

    return transformation.dot_mat(&rotate_mat);
}

pub fn scale(transformation: &Mat4, scale_factor: f32) -> Mat4 {
    let mut scale_mat = Mat4::identity();
    scale_mat.scalar_mul_(1. / scale_factor);
    scale_mat._set_entry(3, 3, 1.);
    return scale_mat.dot_mat(transformation);
}
