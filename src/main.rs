use std::collections::{HashMap, HashSet};
use crate::data::{Vec3, Add, Normalize, Minus, Cross, ScalarMul};
use std::collections::hash_map::RandomState;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

mod err;
mod data;
mod state;

const OBJ_PATH: &'static str = "data/triangle.obj";

fn main() {
    let (mut model, _) = tobj::load_obj(OBJ_PATH, true).expect("Loading Error");
    println!("{}", model.len());
    let mesh = &mut model.get_mut(0).unwrap().mesh;
    println!("{}", mesh.normals.len());
    let mut positions_os = Vec::<Vec3>::new();
    let mut map = HashMap::<u32, Vec<(u32, u32)>>::new();
    println!("triangle num = {}", mesh.num_face_indices.len());
    println!("indices len = {}", mesh.indices.len());
    println!("vertex num = {}", mesh.positions.len() / 3);

    // get positions
    for i in (0..mesh.positions.len()).step_by(3)
    {
        unsafe
            {
                let x = *mesh.positions.get_unchecked(i);
                let y = *mesh.positions.get_unchecked(i + 1);
                let z = *mesh.positions.get_unchecked(i + 2);
                let p = Vec3::new_xyz(x, y, z);
                positions_os.push(p);
            }
    }

    // get adj vertices
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

    let mut normals_os = Vec::new();
    let normals_os = Arc::new(Mutex::new(normals_os));
    let positions_os = &positions_os;
    map.par_iter().for_each(|(vertex, adj_point_vertices)| {
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
            let mut normals_os = normals_os.lock().unwrap();
            normals_os.push((*vertex, vn));
        }
    });
    println!("OK");
}
