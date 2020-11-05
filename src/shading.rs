use std::cmp::{max, min};

use crate::data::{Add, Mat4, MatVecDot, Minus, ScalarMul, Vec3, Vec4, VecDot};
use crate::transformations::{inverse_look_at, look_at};

pub struct Camera
{
    pub pos_wc: Vec3,
    pub gaze_center_wc: Vec3,
    pub up_wc: Vec3,
    pub transformation: Mat4,
    pub inverse_transformation: Mat4,
}

impl Camera
{
    pub fn new(pos_wc: Vec3, gaze_center_wc: Vec3, up_wc: Vec3) -> Self
    {
        Camera
        {
            pos_wc,
            gaze_center_wc,
            up_wc,
            transformation: look_at(&pos_wc, &gaze_center_wc, &up_wc),
            inverse_transformation: inverse_look_at(&pos_wc, &gaze_center_wc, &up_wc),
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Vertex
{
    pub position: Vec4,
    pub idx: usize,
}

impl Vertex
{
    pub fn x(&self) -> f32
    {
        self.position.x()
    }
    pub fn y(&self) -> f32
    {
        self.position.y()
    }
    pub fn z(&self) -> f32
    {
        self.position.z()
    }
    pub fn w(&self) -> f32
    {
        self.position.w()
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Normal
{
    pub vec: Vec4,
    pub vertex_idx: usize,
}

pub struct Triangle<'a>
{
    v1: &'a Vertex,
    v2: &'a Vertex,
    v3: &'a Vertex,
    n1: &'a Normal,
    n2: &'a Normal,
    n3: &'a Normal,
}

impl<'a> Triangle<'a>
{
    pub fn new(vn1: (&'a Vertex, &'a Normal), vn2: (&'a Vertex, &'a Normal), vn3: (&'a Vertex, &'a Normal)) -> Self
    {
        Triangle
        {
            v1: vn1.0,
            n1: vn1.1,
            v2: vn2.0,
            n2: vn2.1,
            v3: vn3.0,
            n3: vn3.1,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Fragment
{
    pub x: u32,
    pub y: u32,
    pub z: f32,
    pub color: Vec3,
}


fn get_min_max(a: f32, b: f32, c: f32, upper_bound: f32, lower_bound: f32) -> (u32, u32)
{
    let mut min = f32::round(f32::min(f32::min(a, b), c));
    let mut max = f32::round(f32::max(f32::max(a, b), c));
    if min < lower_bound
    {
        min = lower_bound;
    }
    if max > upper_bound
    {
        max = upper_bound;
    }
    return (min as u32, max as u32);
}

pub fn rasterization(triangles_ec: &Vec<Triangle>, perspective_mat: &Mat4, width: u32, height: u32) -> Vec<Fragment>
{
    let w_f = width as f32;
    let h_f = height as f32;
    let mut fragments = Vec::<Fragment>::new();
    for triangle_ec in triangles_ec.iter()
    {
        let vs = vec![&triangle_ec.v1.position, &triangle_ec.v2.position, &triangle_ec.v3.position];
        let mut vs_dc: Vec<Vec4> = Vec::new();
        for p in vs.iter()
        {
            let mut v_sc = perspective_mat.mat_vec_dot(*p);
            v_sc.scalar_mul_(1.0 / v_sc.w()); //normalize self
            let v_dc = Vec4::new_xyzw((v_sc.x() + 1.0) * 0.5 * w_f,
                                      (v_sc.y() + 1.0) * 0.5 * h_f,
                                      -1.0 / v_sc.z(), // for perspective correctness, precompute 1/z
                                      1.0);
            vs_dc.push(v_dc);
        }
        // let vs_dc: Vec<Vec4> = vs.iter().map(|p| {
        //     let mut v_sc = perspective_mat.mat_vec_dot(*p);
        //     v_sc.scalar_mul_(1.0 / v_sc.w()); //normalize self
        //     let v_dc = Vec4::new_xyzw((v_sc.x() + 1.0) * 0.5 * w_f,
        //                               (v_sc.y() + 1.0) * 0.5 * h_f,
        //                               1.0 / v_sc.z(), // for perspective correctness, precompute 1/z // TODO: need negate it ?
        //                               1.0);
        //     return v_dc;
        // }).collect();

        let v0_dc = vs_dc.get(0).unwrap();
        let v1_dc = vs_dc.get(1).unwrap();
        let v2_dc = vs_dc.get(2).unwrap();
        let c0 = Vec3::new_rgb(255.0, 0.0, 0.0);
        let c1 = Vec3::new_rgb(0.0, 255.0, 0.0);
        let c2 = Vec3::new_rgb(0.0, 0.0, 255.0);
        let area = triangle_area(v0_dc, v1_dc, v2_dc);

        let (x_min, x_max) = get_min_max(v0_dc.x(), v1_dc.x(), v2_dc.x(), w_f, 0.0);
        let (y_min, y_max) = get_min_max(v0_dc.y(), v1_dc.y(), v2_dc.y(), h_f, 0.0);

        for i in x_min..x_max
        {
            for j in y_min..y_max
            {
                let p = Vec4::new_xyzw((i as f32) + 0.5,
                                       // (height - j) as f32 + 0.5, //seems weird
                                       j as f32 + 0.5, //seems weird
                                       0.0, 0.0);
                let mut w0 = triangle_area(v1_dc, v2_dc, &p);
                let mut w1 = triangle_area(v2_dc, v0_dc, &p);
                let mut w2 = triangle_area(v0_dc, v1_dc, &p);
                if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0
                {
                    w0 /= area;
                    w1 /= area;
                    w2 /= area;
                    let z = 1.0 / (w0 * v0_dc.z() + w1 * v1_dc.z() + w2 * v2_dc.z());
                    let mut color = c0.scalar_mul(w0);
                    color.add_(&c1.scalar_mul(w1));
                    color.add_(&c2.scalar_mul(w2));
                    color.scalar_mul_(z);
                    let f = Fragment {
                        x: i,
                        y: j,
                        z,
                        color,
                    };
                    fragments.push(f);
                }
            }
        }
    }
    return fragments;
}

pub fn raster(triangle_sc: &Triangle) -> Vec<Fragment>
{
    let color = Vec3::new_rgb(255.0, 0.0, 0.0);
    let mut fragments = Vec::<Fragment>::new();
    if triangle_sc.v1.y() == triangle_sc.v2.y() && triangle_sc.v1.y() == triangle_sc.v3.y()
    {
        return fragments;
    } else {
        let mut v = vec![triangle_sc.v1, triangle_sc.v2, triangle_sc.v3];
        v.sort_by(|a, b| (*a).position.y().partial_cmp(&(*b).position.y()).unwrap());
        let y_min = (v.get(0).unwrap().y()) as i32;
        let y_mid = (v.get(1).unwrap().y()) as i32;
        let y_max = (v.get(2).unwrap().y()) as i32;

        let x_min = (v.get(0).unwrap().x()) as i32;
        let x_mid = (v.get(1).unwrap().x()) as i32;
        let x_max = (v.get(2).unwrap().x()) as i32;

        let y_height = y_max - y_min;
        let y_height_f = y_height as f32;
        for i in 0..y_height
        {
            let half = i > y_mid - y_min || y_mid == y_min;
            let seg_height = match half {
                true => y_max - y_mid,
                false => y_mid - y_min
            };
            let alpha = (i as f32) / y_height_f;
            let beta = ((i as f32) - if half { (y_mid - y_min) as f32 } else { 0.0_f32 }) / (seg_height as f32);
            let ax = x_min + ((x_max - x_min) as f32 * alpha) as i32;
            let bx = match half {
                true => x_mid + ((x_max - x_mid) as f32 * beta) as i32,
                false => x_min + ((x_mid - x_min) as f32 * beta) as i32
            };
            let (x_start, x_stop) = if ax > bx { (bx, ax) } else { (ax, bx) };
            for j in x_start..(x_stop + 1)
            {
                fragments.push(Fragment {
                    x: j as u32,
                    y: (y_min + i) as u32,
                    z: 0.0, //TODO: interpolate z
                    color,
                })
            }
        }
        return fragments;
    }
}

pub fn triangle_area(a: &Vec4, b: &Vec4, c: &Vec4) -> f32
{
    let area = (c.x() - a.x()) * (b.y() - a.y()) - (c.y() - a.y()) * (b.x() - a.x());
    return area;
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_area() {
        let v1 = Vec4::new_xyzw(1., 0., 0., 0.0);
        let v2 = Vec4::new_xyzw(0., 1., 0., 0.0);
        let v3 = Vec4::new(0.0);
        println!("{}", triangle_area(&v1, &v2, &v3));
    }
}