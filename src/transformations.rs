use crate::data::{Add, Mat4, Normalize, ScalarMul, Vec3, Vec4, _Mat, Minus, Cross, Mat3};

pub fn translate_obj(mat: Mat4, translation: &Vec3) -> Mat4 {
    let mut result = mat.clone();
    // let translation = translation.scalar_mul(-1.);
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

// Reference: https://en.wikipedia.org/wiki/Rotation_matrix
pub fn rotate_obj(transformation: Mat4, angle: f32, mut axis: Vec3) -> Mat4 {
    // let angle = -angle;
    let cos = angle.cos();
    let one_cos = 1. - cos;
    let sin = angle.sin();
    axis.normalize_();
    let x = axis.x();
    let y = axis.y();
    let z = axis.z();
    let mut rotate_mat = Mat4::identity();
    //first row
    rotate_mat._set_entry(0, 0, cos + one_cos * x * x);
    rotate_mat._set_entry(0, 1, one_cos * x * y - sin * z);
    rotate_mat._set_entry(0, 2, one_cos * x * z + sin * y);
    // second row
    rotate_mat._set_entry(1, 0, one_cos * y * x + sin * z);
    rotate_mat._set_entry(1, 1, cos + one_cos * y * y);
    rotate_mat._set_entry(1, 2, one_cos * y * z - sin * x);
    // third row
    rotate_mat._set_entry(2, 0, one_cos * z * x - sin * y);
    rotate_mat._set_entry(2, 1, one_cos * z * y + sin * x);
    rotate_mat._set_entry(2, 2, cos + one_cos * z * z);
    return transformation.dot_mat(&rotate_mat);
}

pub fn scale(transformation: &Mat4, scale_factor: f32) -> Mat4 {
    let mut scale_mat = Mat4::identity();
    // scale_mat.scalar_mul_(1. / scale_factor);
    scale_mat.scalar_mul_(scale_factor);
    scale_mat._set_entry(3, 3, 1.);
    return scale_mat.dot_mat(transformation);
}

// TODO: check this
pub fn look_at(eye: &Vec3, center: &Vec3, up: &Vec3) -> Mat4 {
    // let mut look_at_direction = center._minus(eye); // left-hand coord
    let mut look_at_direction = eye._minus(center); // right-hand coord. looking at negative z
    look_at_direction.normalize_();
    let mut right = look_at_direction.cross(up);
    right.normalize_();
    let mut camera_up = right.cross(&look_at_direction);
    camera_up.normalize_();
    let f = look_at_direction;
    let mut m = Mat4::identity();

    // m._set_column(0, &right);
    // m._set_column(1, &camera_up);
    // m._set_column(2, &f);
    m._set_row(0, &Vec4::from(&right, 0.0));
    m._set_row(1, &Vec4::from(&camera_up, 0.0));
    m._set_row(2, &Vec4::from(&f, 0.0));

    return m;
}
