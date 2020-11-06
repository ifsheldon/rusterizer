use std::collections::HashMap;
use std::ops::IndexMut;
use std::time::Instant;

use pixel_canvas::{Canvas, Color, XY};
use rayon::prelude::*;
use tobj::Mesh;

use crate::data::{Add, Cross, Mat4, MatVecDot, Minus, Normalize, ScalarDiv, ScalarMul, Transpose, Vec3, Vec4};
use crate::shading::*;
use crate::state::KeyboardMouseStates;
use crate::transformations::perspective;

mod err;
mod data;
mod state;
mod transformations;
mod shading;

const OBJ_PATH: &'static str = "data/triangle_test_dc.obj";
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
                    idx: vertex_idx,
                };
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

pub fn get_triangles<'a>(vertices: &'a Vec<Vertex>, normals: &'a Vec<Normal>, mesh: &Mesh) -> Vec<Triangle<'a>>
{
    let idxs: Vec<usize> = (0..mesh.indices.len()).step_by(3).collect();
    let triangles: Vec<Triangle> = idxs.par_iter().map(|i| {
        let i = *i;
        unsafe {
            let idx1 = (*mesh.indices.get_unchecked(i)) as usize;
            let idx2 = (*mesh.indices.get_unchecked(i + 1)) as usize;
            let idx3 = (*mesh.indices.get_unchecked(i + 2)) as usize;
            let triangle = Triangle::new((vertices.get_unchecked(idx1), normals.get_unchecked(idx1)),
                                         (vertices.get_unchecked(idx2), normals.get_unchecked(idx2)),
                                         (vertices.get_unchecked(idx3), normals.get_unchecked(idx3)));
            return triangle;
        }
    }).collect();
    return triangles;
}

pub fn get_normals(vertices: &Vec<Vertex>, adj_vertices_map: &HashMap<usize, Vec<(usize, usize)>>) -> Vec<Normal>
{
    let mut normals: Vec<Normal> = adj_vertices_map.par_iter().map(|(vertex, adj_point_vertices)| {
        unsafe {
            let mut v_p = vertices.get_unchecked(*vertex).position.clone();
            v_p.scalar_div_(v_p.w());
            let v_p = Vec3::from(&v_p);
            let mut vn = Vec3::new(0.0);
            for adj_vertices in adj_point_vertices.iter()
            {
                let mut v1_p = vertices.get_unchecked(adj_vertices.0).position.clone();
                let mut v2_p = vertices.get_unchecked(adj_vertices.1).position.clone();
                v1_p.scalar_div_(v1_p.w());
                v2_p.scalar_div_(v2_p.w());

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
                vec: Vec4::from(&vn, 0.0),
            };
        }
    }).collect();
    normals.sort_by(|a, b| a.vertex_idx.partial_cmp(&b.vertex_idx).unwrap());
    return normals;
}

const WIDTH: usize = 400;
const HEIGHT: usize = 400;

fn main() {
    let (mut models, _) = tobj::load_obj("data/KAUST_Beacon.obj", true).expect("Loading Error");
    println!("model num = {}", models.len());
    let model = models.pop().unwrap();
    let mesh = model.mesh;
    println!("normal num = {}", mesh.normals.len());
    println!("triangle num = {}", mesh.num_face_indices.len());
    println!("indices len = {}", mesh.indices.len());
    println!("vertex num = {}", mesh.positions.len() / 3);

    let vertices_os = get_position_os(&mesh);
    let adj_vertices_map = get_adj_vertices(&mesh);
    let identity = Mat4::identity();
    let obj_translation = Vec3::new_xyz(-110.0, -90.0, -130.0);
    let obj_os_to_wc_transformation = transformations::translate_obj(&identity, &obj_translation);
    let vertices_wc: Vec<Vertex> = vertices_os.par_iter().map(|v_os| Vertex {
        position: obj_os_to_wc_transformation.mat_vec_dot(&v_os.position),
        idx: v_os.idx,
    }).collect();
    let normals_wc: Vec<Normal> = get_normals(&vertices_wc, &adj_vertices_map);

    let light_pos_wc = Vec4::new_xyzw(200.0, 200.0, 200.0, 1.0);
    let silver_material = Material {
        ambient: Vec3::new_rgb(0.1, 0.1, 0.1),
        diffuse: Vec3::new_rgb(0.5, 0.5, 0.5),
        reflection: Vec3::new_rgb(1.0, 1.0, 1.0),
        global_reflection: Vec3::new_rgb(0.5, 0.5, 0.5),
        specular: 16.0,
    };

    let mut zbuff = ZBuffer::new(WIDTH, HEIGHT, f32::MAX);
    let gouraud_shading = true;

    let canvas = Canvas::new(WIDTH, HEIGHT)
        .title("Rusterizer")
        .state(KeyboardMouseStates::new())
        .input(KeyboardMouseStates::handle_input);

    let now = Instant::now();

    let mut cam_pos_wc = Vec3::new_xyz(0.0, 0.0, 200.0);
    canvas.render(move |state, frame_buffer_image| {
        if state.received_keycode
        {}
        state.reset_flags();
        let camera = Camera::new(cam_pos_wc,
                                 Vec3::new_xyz(0.0, 0.0, 0.0),
                                 Vec3::new_xyz(0.0, 1.0, 0.0));
        let mut light_pos_ec = camera.transformation.mat_vec_dot(&light_pos_wc);
        light_pos_ec.scalar_div_(light_pos_ec.w());
        let normal_mat = camera.inverse_transformation.transpose();
        let mut vertices_ec: Vec<Vertex> = vertices_wc.par_iter().map(|v_wc| {
            let mut p_ec = camera.transformation.mat_vec_dot(&v_wc.position);
            p_ec.scalar_div_(p_ec.w());
            return Vertex {
                position: p_ec,
                idx: v_wc.idx,
            };
        }).collect();
        vertices_ec.sort_by(|a, b| a.idx.partial_cmp(&b.idx).unwrap());
        let mut normal_ec: Vec<Normal> = normals_wc.par_iter().map(|n_wc| {
            let mut n_ec = normal_mat.mat_vec_dot(&n_wc.vec);
            n_ec.normalize_();
            return Normal {
                vertex_idx: n_wc.vertex_idx,
                vec: n_ec,
            };
        }).collect();
        normal_ec.sort_by(|a, b| a.vertex_idx.partial_cmp(&b.vertex_idx).unwrap());

        let light_ec;
        let before_rasterization = now.elapsed().as_millis();
        let mut fragments;
        if gouraud_shading
        {
            light_ec = Light {
                position: Vec3::from(&light_pos_ec),
                original_position: Vec3::from(&light_pos_ec),
                ambient: Vec3::new_rgb(1.0, 1.0, 1.0),
                diffuse: Vec3::new_rgb(1.0, 1.0, 1.0),
            };
            let vertices_colors = gouraud_shade(&vertices_ec, &normal_ec, &light_ec, &silver_material);
            let triangles_ec = get_triangles(&vertices_ec, &vertices_colors, &mesh);
            let proj_mat = perspective(90_f32.to_radians(), (WIDTH as f32) / (HEIGHT as f32), 0.1, 500.0);
            fragments = rasterization(&triangles_ec, &proj_mat, WIDTH as u32, HEIGHT as u32);
        } else {
            light_ec = Light {
                position: Vec3::from(&light_pos_ec),
                original_position: Vec3::from(&light_pos_ec),
                ambient: Vec3::new_rgb(0.3, 0.3, 0.3),
                diffuse: Vec3::new_rgb(0.7, 0.7, 0.7),
            };
            let triangles_ec = get_triangles(&vertices_ec, &normal_ec, &mesh);
            let proj_mat = perspective(90_f32.to_radians(), (WIDTH as f32) / (HEIGHT as f32), 0.1, 500.0);
            fragments = rasterization(&triangles_ec, &proj_mat, WIDTH as u32, HEIGHT as u32);
        }
        let mut survived_fragments = Vec::new();
        while !fragments.is_empty()
        {
            let f = fragments.pop().unwrap();
            if zbuff.update(f.x as usize, f.y as usize, f.z) {
                survived_fragments.push(f);
            }
        }
        let after_rasterization = now.elapsed().as_millis();
        println!("Rasterization took {} ms", after_rasterization - before_rasterization);

        let before_shading = now.elapsed().as_millis();
        let colors: Vec<(XY, Color)> = survived_fragments.par_iter().map(|f| {
            let color = match gouraud_shading {
                true => get_gouraud_color(f),
                false => shade(f, &light_ec, &silver_material)
            };
            return (XY(f.x as usize, f.y as usize), color);
        }).collect();

        for color in colors.iter()
        {
            let xy = &color.0;
            *frame_buffer_image.index_mut(XY(xy.0, xy.1)) = color.1.clone();
        }
        let after_shading = now.elapsed().as_millis();
        println!("Phong shading used {} ms", after_shading - before_shading);
        zbuff.reset(f32::MAX);
    });
    println!("OK");
}

pub struct ZBuffer
{
    depth_buffer: Vec<Vec<f32>>,
}

impl ZBuffer
{
    pub fn new(width: usize, height: usize, init_val: f32) -> Self
    {
        let depth_buffer: Vec<Vec<f32>> = (0..width).map(|_| {
            let col: Vec<f32> = (0..height).map(|_| init_val).collect();
            return col;
        }).collect();
        ZBuffer {
            depth_buffer
        }
    }

    pub fn reset(&mut self, val: f32) {
        self.depth_buffer.iter_mut().for_each(|col| col.iter_mut().for_each(|depth| *depth = val));
    }

    pub fn update(&mut self, x: usize, y: usize, val: f32) -> bool
    {
        let old = self._get(x, y);
        return if *old > val
        {
            *old = val;
            true
        } else {
            false
        };
    }

    pub fn get(&self, x: usize, y: usize) -> f32
    {
        unsafe {
            return *self.depth_buffer.get_unchecked(x).get_unchecked(y);
        }
    }

    #[inline]
    fn _get(&mut self, x: usize, y: usize) -> &mut f32
    {
        unsafe {
            return self.depth_buffer.get_unchecked_mut(x).get_unchecked_mut(y);
        }
    }
}

pub fn get_gouraud_color(fragment: &Fragment) -> Color
{
    // println!("{:?}", fragment.normal_ec);
    let mut color_f = Vec3::from(&fragment.normal_ec);
    color_f.scalar_mul_(-1.0);
    return to_color(color_f);
}

pub fn gouraud_shade(vertices_ec: &Vec<Vertex>, normals_ec: &Vec<Normal>, light: &Light, material: &Material) -> Vec<Normal>
{
    assert_eq!(vertices_ec.len(), normals_ec.len());
    let idxs: Vec<usize> = (0..vertices_ec.len()).collect();
    let mut vertices_colors: Vec<Normal> = idxs.par_iter().map(|i| {
        let i = *i;
        unsafe {
            let n = normals_ec.get_unchecked(i);
            let v = vertices_ec.get_unchecked(i);
            let mut normal_ec = Vec3::from(&n.vec);
            normal_ec.normalize_();
            let pos_ec = v.position.clone();
            let mut light_dir = light.position._minus(&Vec3::from(&pos_ec));
            light_dir.normalize_();
            let mut view_dir = Vec3::from(&pos_ec);
            view_dir.scalar_mul_(-1.0);
            view_dir.normalize_();
            let color_f = phong_lighting(&light_dir, &normal_ec, &view_dir, material, light);
            return Normal {
                vec: Vec4::from(&color_f, 0.0),
                vertex_idx: i,
            };
        }
    }).collect();
    vertices_colors.sort_by(|a, b| a.vertex_idx.partial_cmp(&b.vertex_idx).unwrap());
    return vertices_colors;
}

pub fn shade(fragment: &Fragment, light: &Light, material: &Material) -> Color
{
    let mut normal_ec = Vec3::from(&fragment.normal_ec);
    normal_ec.normalize_();
    let pos_ec = fragment.coord_ec.clone();
    let mut light_dir = light.position._minus(&Vec3::from(&pos_ec));
    light_dir.normalize_();
    let mut view_dir = Vec3::from(&pos_ec);
    view_dir.scalar_mul_(-1.0);
    view_dir.normalize_();
    let color_f = phong_lighting(&light_dir, &normal_ec, &view_dir, material, light);
    return to_color(color_f);
}

#[inline]
fn to_color(mut color: Vec3) -> Color {
    clamp_(&mut color);
    color.scalar_mul_(255.);
    let x = color.r().round();
    let y = color.g().round();
    let z = color.b().round();
    Color::rgb(x as u8, y as u8, z as u8)
}

#[inline]
fn clamp_(color: &mut Vec3) {
    color.set_r(clamp_float(color.r()));
    color.set_g(clamp_float(color.g()));
    color.set_b(clamp_float(color.b()));
}

#[inline]
fn clamp_float(x: f32) -> f32 {
    if x < 0. {
        return 0.;
    }
    if x > 1. {
        return 1.;
    }
    x
}
