use crate::data::{Vec3, Mat4, Vec4};
use crate::transformations::{look_at, inverse_look_at};

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

pub struct Normal
{
    pub vec: Vec4,
    pub vertex_idx: usize
}