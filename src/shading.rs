use crate::data::{Vec3, Mat4, Vec4};
use crate::transformations::{look_at, inverse_look_at};
use std::cmp::{max, min};

pub struct Camera
{
    pub pos_wc: Vec3,
    pub gaze_center_wc: Vec3,
    pub up_wc: Vec3,
    pub transformation: Mat4,
    pub inverse_transformation: Mat4
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
            inverse_transformation: inverse_look_at(&pos_wc, &gaze_center_wc, &up_wc)
        }
    }
}

pub struct Vertex
{
    pub position: Vec4,
    pub idx: usize
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

pub struct Normal
{
    pub vec: Vec4,
    pub vertex_idx: usize
}

pub struct Triangle<'a>
{
    v1: &'a Vertex,
    v2: &'a Vertex,
    v3: &'a Vertex,
    n1: &'a Normal,
    n2: &'a Normal,
    n3: &'a Normal
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
            n3: vn3.1
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Fragment
{
    pub x: u32,
    pub y: u32,
    pub z: f32,
    pub color: Vec3
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
                    color
                })
            }
        }
        return fragments;
    }
}