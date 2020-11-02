mod err;
mod data;
mod state;

const OBJ_PATH: &'static str = "data/KAUST_Beacon.obj";
fn main() {
    let (model, _) = tobj::load_obj(OBJ_PATH, true).expect("Loading Error");
    println!("{}",model.len());
    let mesh = &model.get(0).unwrap().mesh;
}
