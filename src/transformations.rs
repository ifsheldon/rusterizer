use crate::data::{Add, Mat4, Normalize, ScalarMul, Vec3, Vec4, _Mat, Minus, Cross, Mat3};

///
/// Combines Translate Matrix and mat
///
/// return = left_mat dot Translate Matrix
///
pub fn translate_obj(left_mat: &Mat4, translation: &Vec3) -> Mat4 {
    let mut result = left_mat.clone();
    let m0 = left_mat._get_column(0);
    let m1 = left_mat._get_column(1);
    let m2 = left_mat._get_column(2);
    let m3 = left_mat._get_column(3);
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
pub fn rotate_obj(transformation: &Mat4, angle_radian: f32, axis: &Vec3) -> Mat4 {
    // let angle = -angle;
    let cos = angle_radian.cos();
    let one_cos = 1. - cos;
    let sin = angle_radian.sin();
    let mut axis = axis.clone();
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

pub fn inverse_look_at(eye: &Vec3, center: &Vec3, up: &Vec3) -> Mat4
{
    let mut look_at_direction = eye._minus(center); // right-hand coord. looking at negative z
    look_at_direction.normalize_();
    let mut right = look_at_direction.cross(up);
    right.normalize_();
    let mut camera_up = right.cross(&look_at_direction);
    camera_up.normalize_();
    let f = look_at_direction;
    let mut m = Mat4::identity();

    m._set_column(0, &Vec4::from(&right, 0.0));
    m._set_column(1, &Vec4::from(&camera_up, 0.0));
    m._set_column(2, &Vec4::from(&f, 0.0));

    let translate_mat = translate_obj(&Mat4::identity(), &eye);

    return translate_mat.dot_mat(&m);
}

pub fn look_at(eye: &Vec3, center: &Vec3, up: &Vec3) -> Mat4 {
    let mut look_at_direction = eye._minus(center); // right-hand coord. looking at negative z
    look_at_direction.normalize_();
    let mut right = look_at_direction.cross(up);
    right.normalize_();
    let mut camera_up = right.cross(&look_at_direction);
    camera_up.normalize_();
    let f = look_at_direction;
    let mut m = Mat4::identity();

    m._set_row(0, &Vec4::from(&right, 0.0));
    m._set_row(1, &Vec4::from(&camera_up, 0.0));
    m._set_row(2, &Vec4::from(&f, 0.0));

    m = translate_obj(&m, &eye.scalar_mul(-1.0));
    return m;
}


#[cfg(test)]
mod test {
    use super::*;
    use crate::data::MatVecDot;

    #[test]
    fn test_look_at1() {
        let eye_wc = Vec3::new_xyz(0.0, 0.0, 1.);
        let center_wc = Vec3::new_xyz(0.0, 0.0, 0.0);
        let up_wc = Vec3::new_xyz(0.0, 1.0, 0.0);
        let look_at_mat = look_at(&eye_wc, &center_wc, &up_wc);
        let p_wc = Vec4::new_xyzw(0.0, 0.0, 0.0, 1.0);
        let p_ec = look_at_mat.mat_vec_dot(&p_wc);
        assert_eq!(p_ec.x(), 0.0);
        assert_eq!(p_ec.y(), 0.0);
        assert_eq!(p_ec.z(), -1.0);
        assert_eq!(p_wc.w(), 1.0);
        println!("{:?}", p_ec);
    }

    #[test]
    fn test_look_at2() {
        let eye_wc = Vec3::new_xyz(0.0, 3.0, 4.0);
        let center_wc = Vec3::new_xyz(0.0, 0.0, 0.0);
        let up_wc = Vec3::new_xyz(0.0, 1.0, 0.0);
        let look_at_mat = look_at(&eye_wc, &center_wc, &up_wc);
        let p_wc = Vec4::new_xyzw(0.0, 0.0, 0.0, 1.0);
        let p_ec = look_at_mat.mat_vec_dot(&p_wc);
        assert_eq!(p_ec.x(), 0.0);
        assert_eq!(p_ec.y(), 0.0);
        assert_eq!(p_ec.z(), -5.0);
        assert_eq!(p_wc.w(), 1.0);
        println!("{:?}", p_ec);
    }

    #[test]
    fn test_look_at3() {
        let eye_wc = Vec3::new_xyz(4.0, 3.0, 0.0);
        let center_wc = Vec3::new_xyz(0.0, 0.0, 0.0);
        let up_wc = Vec3::new_xyz(0.0, 1.0, 0.0);
        let look_at_mat = look_at(&eye_wc, &center_wc, &up_wc);
        let p_wc = Vec4::new_xyzw(0.0, 0.0, 0.0, 1.0);
        let p_ec = look_at_mat.mat_vec_dot(&p_wc);
        assert_eq!(p_ec.x(), 0.0);
        assert_eq!(p_ec.y(), 0.0);
        assert_eq!(p_ec.z(), -5.0);
        assert_eq!(p_wc.w(), 1.0);
        println!("{:?}", p_ec);
    }

    #[test]
    fn test_look_at4() {
        let eye_wc = Vec3::new_xyz(4.0, 0.0, 3.0);
        let center_wc = Vec3::new_xyz(0.0, 0.0, 0.0);
        let up_wc = Vec3::new_xyz(0.0, 1.0, 0.0);
        let look_at_mat = look_at(&eye_wc, &center_wc, &up_wc);
        let p_wc = Vec4::new_xyzw(0.0, 0.0, 0.0, 1.0);
        let p_ec = look_at_mat.mat_vec_dot(&p_wc);
        assert_eq!(p_ec.x(), 0.0);
        assert_eq!(p_ec.y(), 0.0);
        assert_eq!(p_ec.z(), -5.0);
        assert_eq!(p_wc.w(), 1.0);
        println!("{:?}", p_ec);
    }

    #[test]
    fn test_inverse_look_at()
    {
        let eye_wc = Vec3::new_xyz(4.0, 0.0, 3.0);
        let center_wc = Vec3::new_xyz(0.0, 0.0, 0.0);
        let up_wc = Vec3::new_xyz(0.0, 1.0, 0.0);
        let look_at_mat = look_at(&eye_wc, &center_wc, &up_wc);
        let p_wc = Vec4::new_xyzw(30.0, 100.0, 100.0, 1.0);
        let p_ec = look_at_mat.mat_vec_dot(&p_wc);
        let inverse_look_at_mat = inverse_look_at(&eye_wc, &center_wc, &up_wc);
        let p_ec_to_wc = inverse_look_at_mat.mat_vec_dot(&p_ec);
        // may fail due to precision problems
        assert_eq!(p_wc.x(), p_ec_to_wc.x());
        assert_eq!(p_wc.y(), p_ec_to_wc.y());
        assert_eq!(p_wc.z(), p_ec_to_wc.z());
    }

    #[test]
    fn test_rotate()
    {
        let x_axis = Vec3::new_xyz(1.0, 0.0, 0.0);
        let y_axis = Vec3::new_xyz(0.0, 1.0, 0.0);
        let z_axis = Vec3::new_xyz(0.0, 0.0, 1.0);
        let p = Vec4::new_xyzw(1.0, 1.0, 1.0, 1.0);
        let rad_90 = (90.0_f32).to_radians();
        let identity = Mat4::identity();
        let x_90 = rotate_obj(&identity, rad_90, &x_axis);
        let y_90 = rotate_obj(&identity, rad_90, &y_axis);
        let z_90 = rotate_obj(&identity, rad_90, &z_axis);
        let p_x = x_90.mat_vec_dot(&p);
        let p_y = y_90.mat_vec_dot(&p);
        let p_z = z_90.mat_vec_dot(&p);
        println!("{:?}\n{:?}\n{:?}", p_x, p_y, p_z);
    }
}