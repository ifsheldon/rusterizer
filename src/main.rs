use std::collections::{HashMap, HashSet};
use crate::data::{Vec3, Add, Normalize, Minus, Cross, ScalarMul, Mat4, Vec4, MatVecDot};
use std::collections::hash_map::RandomState;
use rayon::prelude::*;
use std::sync::{Arc, Mutex, mpsc};
use tobj::Mesh;

mod err;
mod data;
mod state;
mod transformations;

const OBJ_PATH: &'static str = "data/triangle.obj";
const OBJECT_CENTER: (f32, f32, f32) = (125.0, 125.0, 125.0);
const OBJ_BOUNDING_RADIUS: f32 = 125.0;

pub fn get_position_os(mesh: &Mesh) -> Vec<Vec3>
{
    let idxs: Vec<usize> = (0..mesh.positions.len()).step_by(3).collect();
    let positions_os: Vec<Vec3> = idxs.par_iter().map(|i| {
        let i = *i;
        unsafe
            {
                let x = *mesh.positions.get_unchecked(i);
                let y = *mesh.positions.get_unchecked(i + 1);
                let z = *mesh.positions.get_unchecked(i + 2);
                return Vec3::new_xyz(x, y, z);
            }
    }).collect();
    return positions_os;
}

pub fn get_adj_vertices(mesh: &Mesh) -> HashMap<u32, Vec<(u32, u32)>>
{
    let mut map = HashMap::<u32, Vec<(u32, u32)>>::new();
    for i in (0..mesh.indices.len()).step_by(3)
    {
        unsafe {
            let idx1 = mesh.indices.get_unchecked(i);
            let idx2 = mesh.indices.get_unchecked(i + 1);
            let idx3 = mesh.indices.get_unchecked(i + 2);
            match map.get_mut(idx1)
            {
                None => {
                    let v = vec![(*idx2, *idx3)];
                    map.insert(*idx1, v);
                }
                Some(vec) => {
                    vec.push((*idx2, *idx3));
                }
            }

            match map.get_mut(idx2)
            {
                None => {
                    let v = vec![(*idx3, *idx1)];
                    map.insert(*idx2, v);
                }
                Some(vec) => {
                    vec.push((*idx3, *idx1));
                }
            }

            match map.get_mut(idx3)
            {
                None => {
                    let v = vec![(*idx1, *idx2)];
                    map.insert(*idx3, v);
                }
                Some(vec) => {
                    vec.push((*idx1, *idx2));
                }
            }
        }
    }
    return map;
}

fn main() {
    let (mut model, _) = tobj::load_obj(OBJ_PATH, true).expect("Loading Error");
    println!("model num = {}", model.len());
    let mesh = &mut model.get_mut(0).unwrap().mesh;
    println!("normal num = {}", mesh.normals.len());
    println!("triangle num = {}", mesh.num_face_indices.len());
    println!("indices len = {}", mesh.indices.len());
    println!("vertex num = {}", mesh.positions.len() / 3);

    let mut positions_os = get_position_os(mesh);
    let mut adj_vertices_map = get_adj_vertices(mesh);

    let mut normals_os: Vec<(u32, Vec3)> = adj_vertices_map.par_iter().map(|(vertex, adj_point_vertices)| {
        unsafe {
            let v_p = positions_os.get_unchecked((*vertex) as usize);
            let mut vn = Vec3::new(0.0);
            for adj_vertices in adj_point_vertices.iter()
            {
                let v1_p = positions_os.get_unchecked((adj_vertices.0) as usize);
                let v2_p = positions_os.get_unchecked((adj_vertices.1) as usize);
                let v_v1 = v1_p._minus(v_p);
                let v_v2 = v2_p._minus(v_p);
                let mut n = v_v1.cross(&v_v2);
                n.normalize_();
                vn.add_(&n);
            }
            vn.normalize_();
            return (*vertex, vn);
        }
    }).collect();
    normals_os.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let identity = Mat4::identity();
    let obj_translation = Vec3::new_xyz(0.0, 0.0, 0.0);
    let obj_os_to_wc_transformation = transformations::translate_obj(&identity, &obj_translation);
    let positions_wc: Vec<Vec4> = positions_os.par_iter().map(|p_os| {
        let p = Vec4::from(p_os, 1.0);
        return obj_os_to_wc_transformation.mat_vec_dot(&p);
    }).collect();

    let camera_pos_wc = Vec3::new_xyz(0.0, 0.0, 1.0);
    let camera_up_wc = Vec3::new_xyz(0.0, 1.0, 0.0);
    let camera_focus_center_wc = Vec3::new_xyz(0.0, 0.0, 0.0);
    let camera_lookat_mat = transformations::look_at(&camera_pos_wc, &camera_focus_center_wc, &camera_up_wc);
    let positions_ec: Vec<Vec4> = positions_wc.par_iter().map(|p_wc| {
        let mut p_ec = camera_lookat_mat.mat_vec_dot(p_wc);
        p_ec.scalar_mul_(1.0 / p_ec.w());
        return p_ec;
    }).collect();

    println!("OK");
}
