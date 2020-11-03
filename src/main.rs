use std::collections::{HashMap, HashSet};
use crate::data::{Vec3, Add, Normalize, Minus, Cross, ScalarMul, Mat4, Vec4, MatVecDot};
use std::collections::hash_map::RandomState;
use rayon::prelude::*;
use std::sync::{Arc, Mutex, mpsc};
use tobj::Mesh;
use crate::shading::{Vertex, Normal, Camera};

mod err;
mod data;
mod state;
mod transformations;
mod shading;

const OBJ_PATH: &'static str = "data/triangle.obj";
const OBJECT_CENTER: (f32, f32, f32) = (125.0, 125.0, 125.0);
const OBJ_BOUNDING_RADIUS: f32 = 125.0;

pub fn get_position_os(mesh: &Mesh) -> Vec<Vertex>
{
    let idxs: Vec<usize> = (0..mesh.positions.len()).step_by(3).collect();
    let mut positions_os: Vec<Vertex> = idxs.par_iter().map(|i| {
        let i = *i;
        let vertex_idx = i / 3;
        unsafe
            {
                let x = *mesh.positions.get_unchecked(i);
                let y = *mesh.positions.get_unchecked(i + 1);
                let z = *mesh.positions.get_unchecked(i + 2);
                return Vertex {
                    position: Vec4::new_xyzw(x, y, z, 1.0),
                    idx: vertex_idx
                }
            }
    }).collect();
    positions_os.sort_by(|a, b| a.idx.partial_cmp(&b.idx).unwrap());
    return positions_os;
}

pub fn get_adj_vertices(mesh: &Mesh) -> HashMap<usize, Vec<(usize, usize)>>
{
    let mut map = HashMap::<usize, Vec<(usize, usize)>>::new();
    for i in (0..mesh.indices.len()).step_by(3)
    {
        unsafe {
            let idx1 = (*mesh.indices.get_unchecked(i)) as usize;
            let idx2 = (*mesh.indices.get_unchecked(i + 1)) as usize;
            let idx3 = (*mesh.indices.get_unchecked(i + 2)) as usize;
            match map.get_mut(&idx1)
            {
                None => {
                    let v = vec![(idx2, idx3)];
                    map.insert(idx1, v);
                }
                Some(vec) => {
                    vec.push((idx2, idx3));
                }
            }

            match map.get_mut(&idx2)
            {
                None => {
                    let v = vec![(idx3, idx1)];
                    map.insert(idx2, v);
                }
                Some(vec) => {
                    vec.push((idx3, idx1));
                }
            }

            match map.get_mut(&idx3)
            {
                None => {
                    let v = vec![(idx1, idx2)];
                    map.insert(idx3, v);
                }
                Some(vec) => {
                    vec.push((idx1, idx2));
                }
            }
        }
    }
    return map;
}

pub fn get_normals(vertices: &Vec<Vertex>, adj_vertices_map: &HashMap<usize, Vec<(usize, usize)>>) -> Vec<Normal>
{
    let mut normals: Vec<Normal> = adj_vertices_map.par_iter().map(|(vertex, adj_point_vertices)| {
        unsafe {
            let mut v_p = vertices.get_unchecked(*vertex).position.clone();
            v_p.scalar_mul_(1.0 / v_p.w());
            let v_p = Vec3::from(&v_p);
            let mut vn = Vec3::new(0.0);
            for adj_vertices in adj_point_vertices.iter()
            {
                let mut v1_p = vertices.get_unchecked(adj_vertices.0).position.clone();
                let mut v2_p = vertices.get_unchecked(adj_vertices.1).position.clone();
                v1_p.scalar_mul_(1.0 / v1_p.w());
                v2_p.scalar_mul_(1.0 / v2_p.w());

                let v1_p = Vec3::from(&v1_p);
                let v2_p = Vec3::from(&v2_p);
                let v_v1 = v1_p._minus(&v_p);
                let v_v2 = v2_p._minus(&v_p);

                let mut n = v_v1.cross(&v_v2);
                n.normalize_();
                vn.add_(&n);
            }
            vn.normalize_();
            return Normal {
                vertex_idx: *vertex,
                vec: Vec4::from(&vn, 0.0)
            }
        }
    }).collect();
    normals.sort_by(|a, b| a.vertex_idx.partial_cmp(&b.vertex_idx).unwrap());
    return normals;
}

fn main() {
    let (mut model, _) = tobj::load_obj(OBJ_PATH, true).expect("Loading Error");
    println!("model num = {}", model.len());
    let mesh = &mut model.get_mut(0).unwrap().mesh;
    println!("normal num = {}", mesh.normals.len());
    println!("triangle num = {}", mesh.num_face_indices.len());
    println!("indices len = {}", mesh.indices.len());
    println!("vertex num = {}", mesh.positions.len() / 3);

    let mut vertices_os = get_position_os(mesh);
    let mut adj_vertices_map = get_adj_vertices(mesh);
    let mut normals_os: Vec<Normal> = get_normals(&vertices_os, &adj_vertices_map);

    let identity = Mat4::identity();
    let obj_translation = Vec3::new_xyz(0.0, 0.0, 0.0);
    let obj_os_to_wc_transformation = transformations::translate_obj(&identity, &obj_translation);
    let vertices_wc: Vec<Vertex> = vertices_os.par_iter().map(|v_os| Vertex {
        position: obj_os_to_wc_transformation.mat_vec_dot(&v_os.position),
        idx: v_os.idx
    }).collect();

    let camera = Camera::new(Vec3::new_xyz(0.0, 0.0, 1.0),
                             Vec3::new_xyz(0.0, 0.0, 0.0),
                             Vec3::new_xyz(0.0, 1.0, 0.0));
    let vertices_ec: Vec<Vertex> = vertices_wc.par_iter().map(|v_wc| {
        let mut p_ec = camera.transformation.mat_vec_dot(&v_wc.position);
        p_ec.scalar_mul_(1.0 / p_ec.w());
        return Vertex {
            position: p_ec,
            idx: v_wc.idx
        }
    }).collect();

    println!("OK");
}
