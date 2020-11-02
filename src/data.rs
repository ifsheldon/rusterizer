use crate::err::{DimensionMismatchError, OutOfBoundError};

pub trait VecDot<Rhs = Self> {
    fn dot(&self, rhs: &Rhs) -> f32;
}

pub trait MatVecDot<Rhs: Vec> {
    fn mat_vec_dot(&self, rhs: &Rhs) -> Rhs;
}

pub trait Product<Rhs = Self> {
    fn product(&self, rhs: &Rhs) -> Rhs;
    fn product_(&mut self, rhs: &Rhs);
}

pub trait Mat {
    fn get_entry(&self, row: usize, col: usize) -> Result<f32, OutOfBoundError>;
    fn set_entry(&mut self, row: usize, col: usize, val: f32) -> Result<(), OutOfBoundError>;
    fn get_size(&self) -> [usize; 2];
}

pub(crate) trait _Mat: Mat {
    fn _get_entry(&self, row: usize, col: usize) -> f32;
    fn _set_entry(&mut self, row: usize, col: usize, val: f32);
}

pub(crate) trait _Vec: Vec {
    fn _get(&self, index: usize) -> f32 {
        self.get(index).expect("Should NOT happen")
    }

    fn _set(&mut self, index: usize, val: f32) {
        self.set(index, val).expect("Should NOT happen")
    }
}

// set default implementation for all Vec
impl<T: Vec> _Vec for T {}

pub trait Vec {
    fn get(&self, index: usize) -> Result<f32, OutOfBoundError>;
    fn set(&mut self, index: usize, val: f32) -> Result<(), OutOfBoundError>;
    fn get_size(&self) -> usize;
}

pub trait ScalarMul<Output = Self> {
    fn scalar_mul(&self, s: f32) -> Output;
    fn scalar_mul_(&mut self, s: f32);
}

pub trait Cross<Rhs = Self> {
    fn cross(&self, other: &Rhs) -> Rhs;
}

pub trait Add<Output = Self> {
    fn add(&self, other: &Output) -> Result<Output, DimensionMismatchError>;
    fn add_(&mut self, other: &Output);
    fn _add(&self, other: &Output) -> Output;
}

pub trait Minus<Rhs = Self> {
    fn minus(&self, right: &Rhs) -> Result<Rhs, DimensionMismatchError>;
    fn minus_(&mut self, right: &Rhs);
    fn _minus(&self, right: &Rhs) -> Rhs;
}

pub trait Transpose<Output = Self> {
    fn transpose(&self) -> Output;
    fn transpose_(&mut self);
}

/// A trait enabling vector normalization
/// # Notice
/// The implementation should not worry about zero vector
pub trait Normalize<Output = Self> {
    fn normalize(&self) -> Output;
    fn normalize_(&mut self);
}

pub trait Length {
    fn get_length(&self) -> f32;
}

/// A trait enabling matrix inverse
/// # Notice
/// The implementation should not worry about zero vector
pub trait Inverse<Output = Self> {
    fn inverse(&self) -> Output;
}

// column first storage
#[derive(Clone, Copy)]
pub struct Mat4 {
    pub(self) transposed: bool,
    pub data: [[f32; 4]; 4],
}

impl Mat4 {
    pub fn identity() -> Self {
        let data = [
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ];
        Mat4 {
            transposed: false,
            data,
        }
    }

    pub(crate) fn _new1(transposed: bool, data: [[f32; 4]; 4]) -> Self {
        Mat4 { transposed, data }
    }
    pub(crate) fn _new2(transposed: bool, data: [[f32; 4]; 4]) -> Self {
        Mat4 { transposed, data }
    }

    pub(crate) fn _set_row(&mut self, row: usize, val: &Vec4) {
        self._set_entry(row, 0, val.x());
        self._set_entry(row, 1, val.y());
        self._set_entry(row, 2, val.z());
        self._set_entry(row, 3, val.w());
    }

    pub(crate) fn _get_row(&self, row: usize) -> Vec4 {
        let v = Vec4::new_xyzw(
            self._get_entry(row, 0),
            self._get_entry(row, 1),
            self._get_entry(row, 2),
            self._get_entry(row, 3),
        );
        return v;
    }

    pub(crate) fn _set_column(&mut self, column: usize, val: &Vec4) {
        self._set_entry(0, column, val.x());
        self._set_entry(1, column, val.y());
        self._set_entry(2, column, val.z());
        self._set_entry(3, column, val.w());
    }

    pub(crate) fn _get_column(&self, column: usize) -> Vec4 {
        let v = Vec4::new_xyzw(
            self._get_entry(0, column),
            self._get_entry(1, column),
            self._get_entry(2, column),
            self._get_entry(3, column),
        );
        return v;
    }

    #[inline]
    fn transposed_get(&self, row: usize, col: usize) -> f32 {
        self.data[col][row]
    }
    #[inline]
    fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row][col]
    }

    pub fn dot_mat(&self, other: &Mat4) -> Mat4 {
        let mut prod = [[0.0; 4]; 4];
        let self_get = if self.transposed {
            Mat4::transposed_get
        } else {
            Mat4::get
        };
        let other_get = if other.transposed {
            Mat4::transposed_get
        } else {
            Mat4::get
        };
        let mut entry;
        for row in 0..4 {
            for col in 0..4 {
                entry = 0.0;
                for idx in 0..4 {
                    entry += self_get(self, row, idx) * other_get(other, idx, col);
                }
                prod[row][col] = entry;
            }
        }
        return Mat4 {
            transposed: false,
            data: prod,
        };
    }
}

impl MatVecDot<Vec4> for Mat4 {
    fn mat_vec_dot(&self, rhs: &Vec4) -> Vec4 {
        let v = Vec4::new_xyzw(
            self._get_entry(0, 0) * rhs.x()
                + self._get_entry(0, 1) * rhs.y()
                + self._get_entry(0, 2) * rhs.z()
                + self._get_entry(0, 3) * rhs.w(),
            self._get_entry(1, 0) * rhs.x()
                + self._get_entry(1, 1) * rhs.y()
                + self._get_entry(1, 2) * rhs.z()
                + self._get_entry(1, 3) * rhs.w(),
            self._get_entry(2, 0) * rhs.x()
                + self._get_entry(2, 1) * rhs.y()
                + self._get_entry(2, 2) * rhs.z()
                + self._get_entry(2, 3) * rhs.w(),
            self._get_entry(3, 0) * rhs.x()
                + self._get_entry(3, 1) * rhs.y()
                + self._get_entry(3, 2) * rhs.z()
                + self._get_entry(3, 3) * rhs.w(),
        );
        return v;
    }
}

impl Mat for Mat4 {
    fn get_entry(&self, row: usize, col: usize) -> Result<f32, OutOfBoundError> {
        return if row > 3 || col > 3 {
            Err(OutOfBoundError::new([3, 3], [row, col]))
        } else {
            Ok(if self.transposed {
                self.transposed_get(row, col)
            } else {
                self.get(row, col)
            })
        };
    }

    fn set_entry(&mut self, row: usize, col: usize, val: f32) -> Result<(), OutOfBoundError> {
        return if row >= 4 || col >= 4 {
            Err(OutOfBoundError::new([3, 3], [row, col]))
        } else {
            if self.transposed {
                self.data[col][row] = val;
            } else {
                self.data[row][col] = val;
            }
            Ok(())
        };
    }

    fn get_size(&self) -> [usize; 2] {
        [4, 4]
    }
}

impl _Mat for Mat4 {
    fn _get_entry(&self, row: usize, col: usize) -> f32 {
        if self.transposed {
            self.transposed_get(row, col)
        } else {
            self.get(row, col)
        }
    }

    fn _set_entry(&mut self, row: usize, col: usize, val: f32) {
        if self.transposed {
            self.data[col][row] = val;
        } else {
            self.data[row][col] = val;
        }
    }
}

impl ScalarMul for Mat4 {
    fn scalar_mul(&self, s: f32) -> Self {
        let mut data = self.data.clone();
        data[0][0] *= s;
        data[0][1] *= s;
        data[0][2] *= s;
        data[0][3] *= s;

        data[1][0] *= s;
        data[1][1] *= s;
        data[1][2] *= s;
        data[1][3] *= s;

        data[2][0] *= s;
        data[2][1] *= s;
        data[2][2] *= s;
        data[2][3] *= s;

        data[3][0] *= s;
        data[3][1] *= s;
        data[3][2] *= s;
        data[3][3] *= s;
        Mat4 {
            transposed: self.transposed,
            data,
        }
    }

    fn scalar_mul_(&mut self, s: f32) {
        self.data[0][0] *= s;
        self.data[0][1] *= s;
        self.data[0][2] *= s;
        self.data[0][3] *= s;

        self.data[1][0] *= s;
        self.data[1][1] *= s;
        self.data[1][2] *= s;
        self.data[1][3] *= s;

        self.data[2][0] *= s;
        self.data[2][1] *= s;
        self.data[2][2] *= s;
        self.data[2][3] *= s;

        self.data[3][0] *= s;
        self.data[3][1] *= s;
        self.data[3][2] *= s;
        self.data[3][3] *= s;
    }
}

impl Inverse for Mat4 {
    fn inverse(&self) -> Self {
        unimplemented!()
    }
}

impl Transpose for Mat4 {
    fn transpose(&self) -> Self {
        return Mat4 {
            transposed: !self.transposed,
            data: self.data.clone(),
        };
    }

    fn transpose_(&mut self) {
        self.transposed = !self.transposed;
    }
}

// column first storage
#[derive(Clone, Copy)]
pub struct Mat3 {
    transposed: bool,
    data: [[f32; 3]; 3],
}

impl Mat3 {
    pub fn identity() -> Self {
        let data = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        Mat3 {
            transposed: false,
            data,
        }
    }
    pub(crate) fn _new1(transposed: bool, data: [[f32; 3]; 3]) -> Self {
        Mat3 { transposed, data }
    }
    pub(crate) fn _new2(transposed: bool, data: [[f32; 3]; 3]) -> Self {
        Mat3 { transposed, data }
    }

    pub(crate) fn _set_row(&mut self, row: usize, val: &Vec3) {
        self._set_entry(row, 0, val.x());
        self._set_entry(row, 1, val.y());
        self._set_entry(row, 2, val.z());
    }

    pub(crate) fn _get_row(&self, row: usize) -> Vec3 {
        let v = Vec3::new_xyz(
            self._get_entry(row, 0),
            self._get_entry(row, 1),
            self._get_entry(row, 2),
        );
        return v;
    }

    pub(crate) fn _set_column(&mut self, column: usize, val: &Vec3) {
        self._set_entry(0, column, val.x());
        self._set_entry(1, column, val.y());
        self._set_entry(2, column, val.z());
    }

    pub(crate) fn _get_column(&self, column: usize) -> Vec3 {
        let v = Vec3::new_xyz(
            self._get_entry(0, column),
            self._get_entry(1, column),
            self._get_entry(2, column),
        );
        return v;
    }

    #[inline]
    fn transposed_get(&self, row: usize, col: usize) -> f32 {
        self.data[col][row]
    }

    #[inline]
    fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row][col]
    }

    pub fn dot_mat(&self, other: &Mat3) -> Mat3 {
        let mut prod = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let self_get = if self.transposed {
            Mat3::transposed_get
        } else {
            Mat3::get
        };
        let other_get = if other.transposed {
            Mat3::transposed_get
        } else {
            Mat3::get
        };
        let mut entry;
        for row in 0..3 {
            for col in 0..3 {
                entry = 0.0;
                for idx in 0..3 {
                    entry += self_get(self, row, idx) * other_get(other, idx, col);
                }
                prod[row][col] = entry;
            }
        }
        return Mat3 {
            transposed: false,
            data: prod,
        };
    }
}

impl MatVecDot<Vec3> for Mat3 {
    fn mat_vec_dot(&self, rhs: &Vec3) -> Vec3 {
        let v = Vec3::new_xyz(
            self._get_entry(0, 0) * rhs.x()
                + self._get_entry(0, 1) * rhs.y()
                + self._get_entry(0, 2) * rhs.z(),
            self._get_entry(1, 0) * rhs.x()
                + self._get_entry(1, 1) * rhs.y()
                + self._get_entry(1, 2) * rhs.z(),
            self._get_entry(2, 0) * rhs.x()
                + self._get_entry(2, 1) * rhs.y()
                + self._get_entry(2, 2) * rhs.z(),
        );
        return v;
    }
}

impl Mat for Mat3 {
    fn get_entry(&self, row: usize, col: usize) -> Result<f32, OutOfBoundError> {
        return if row > 2 || col > 2 {
            Err(OutOfBoundError::new([2, 2], [row, col]))
        } else {
            Ok(if self.transposed {
                self.transposed_get(row, col)
            } else {
                self.get(row, col)
            })
        };
    }

    fn set_entry(&mut self, row: usize, col: usize, val: f32) -> Result<(), OutOfBoundError> {
        return if row >= 3 || col >= 3 {
            Err(OutOfBoundError::new([2, 2], [row, col]))
        } else {
            if self.transposed {
                self.data[col][row] = val;
            } else {
                self.data[row][col] = val;
            }
            Ok(())
        };
    }

    fn get_size(&self) -> [usize; 2] {
        [3, 3]
    }
}

impl _Mat for Mat3 {
    fn _get_entry(&self, row: usize, col: usize) -> f32 {
        if self.transposed {
            self.transposed_get(row, col)
        } else {
            self.get(row, col)
        }
    }

    fn _set_entry(&mut self, row: usize, col: usize, val: f32) {
        if self.transposed {
            self.data[col][row] = val;
        } else {
            self.data[row][col] = val;
        }
    }
}

impl ScalarMul for Mat3 {
    fn scalar_mul(&self, s: f32) -> Self {
        let mut data = self.data.clone();
        data[0][0] *= s;
        data[0][1] *= s;
        data[0][2] *= s;

        data[1][0] *= s;
        data[1][1] *= s;
        data[1][2] *= s;

        data[2][0] *= s;
        data[2][1] *= s;
        data[2][2] *= s;

        Mat3 {
            transposed: self.transposed,
            data,
        }
    }

    fn scalar_mul_(&mut self, s: f32) {
        self.data[0][0] *= s;
        self.data[0][1] *= s;
        self.data[0][2] *= s;

        self.data[1][0] *= s;
        self.data[1][1] *= s;
        self.data[1][2] *= s;

        self.data[2][0] *= s;
        self.data[2][1] *= s;
        self.data[2][2] *= s;
    }
}

impl Transpose for Mat3 {
    fn transpose(&self) -> Self {
        return Mat3 {
            transposed: !self.transposed,
            data: self.data.clone(),
        };
    }

    fn transpose_(&mut self) {
        self.transposed = !self.transposed;
    }
}

impl Inverse for Mat3 {
    fn inverse(&self) -> Self {
        unimplemented!()
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Vec3 {
    transposed: bool,
    data: [f32; 3],
}

impl Vec for Vec3 {
    fn get(&self, index: usize) -> Result<f32, OutOfBoundError> {
        return if index > 2 {
            Err(OutOfBoundError::new([2, 0], [index, 0]))
        } else {
            Ok(self.data[index])
        };
    }

    fn set(&mut self, index: usize, val: f32) -> Result<(), OutOfBoundError> {
        return if index > 2 {
            Err(OutOfBoundError::new([2, 0], [index, 0]))
        } else {
            self.data[index] = val;
            Ok(())
        };
    }

    fn get_size(&self) -> usize {
        3
    }
}

impl VecDot for Vec3 {
    fn dot(&self, other: &Self) -> f32 {
        let accum = self.x() * other.x() + self.y() * other.y() + self.z() * other.z();
        return accum;
    }
}

impl Add for Vec3 {
    fn add(&self, other: &Vec3) -> Result<Self, DimensionMismatchError> {
        if self.transposed != other.transposed {
            return Err(DimensionMismatchError::new(
                if self.transposed { [1, 3] } else { [3, 1] },
                if other.transposed { [1, 3] } else { [3, 1] },
            ));
        } else {
            let d = [
                self.x() + other.x(),
                self.y() + other.y(),
                self.z() + other.z(),
            ];
            return Ok(Vec3 {
                data: d,
                transposed: self.transposed,
            });
        }
    }

    fn add_(&mut self, other: &Self) {
        self.data[0] += other.data[0];
        self.data[1] += other.data[1];
        self.data[2] += other.data[2];
    }

    fn _add(&self, v: &Vec3) -> Vec3 {
        let data = [
            self.data[0] + v.data[0],
            self.data[1] + v.data[1],
            self.data[2] + v.data[2],
        ];
        return Vec3 {
            transposed: false,
            data,
        };
    }
}

impl Minus for Vec3 {
    fn minus(&self, right: &Self) -> Result<Self, DimensionMismatchError> {
        if self.transposed != right.transposed {
            return Err(DimensionMismatchError::new(
                if self.transposed { [1, 3] } else { [3, 1] },
                if right.transposed { [1, 3] } else { [3, 1] },
            ));
        } else {
            let d = [
                self.x() - right.x(),
                self.y() - right.y(),
                self.z() - right.z(),
            ];
            return Ok(Vec3 {
                data: d,
                transposed: self.transposed,
            });
        }
    }

    fn minus_(&mut self, right: &Self) {
        self.data[0] -= right.data[0];
        self.data[1] -= right.data[1];
        self.data[2] -= right.data[2];
    }

    fn _minus(&self, right: &Self) -> Self {
        let data = [
            self.data[0] - right.data[0],
            self.data[1] - right.data[1],
            self.data[2] - right.data[2],
        ];
        return Vec3 {
            transposed: false,
            data,
        };
    }
}

impl Cross for Vec3 {
    fn cross(&self, right: &Self) -> Self {
        let d = [
            self.y() * right.z() - self.z() * right.y(),
            self.z() * right.x() - self.x() * right.z(),
            self.x() * right.y() - self.y() * right.x(),
        ];
        Vec3 {
            data: d,
            transposed: false,
        }
    }
}

impl Transpose for Vec3 {
    fn transpose(&self) -> Self {
        let mut v = self.clone();
        v.transposed = !self.transposed;
        return v;
    }

    fn transpose_(&mut self) {
        self.transposed = !self.transposed;
    }
}

impl Length for Vec3 {
    fn get_length(&self) -> f32 {
        let x2 = self.data[0] * self.data[0];
        let y2 = self.data[1] * self.data[1];
        let z2 = self.data[2] * self.data[2];
        let l2 = x2 + y2 + z2;
        return l2.sqrt();
    }
}

impl Product for Vec3 {
    fn product(&self, rhs: &Self) -> Self {
        let v = Vec3::new_xyz(self.x() * rhs.x(), self.y() * rhs.y(), self.z() * rhs.z());
        return v;
    }

    fn product_(&mut self, rhs: &Self) {
        self.data[0] *= rhs.data[0];
        self.data[1] *= rhs.data[1];
        self.data[2] *= rhs.data[2];
    }
}

impl Normalize for Vec3 {
    fn normalize(&self) -> Self {
        let l = self.get_length();
        Vec3::new_xyz(self.data[0] / l, self.data[1] / l, self.data[2] / l)
    }

    fn normalize_(&mut self) {
        let l = self.get_length();
        self.data[0] /= l;
        self.data[1] /= l;
        self.data[2] /= l;
    }
}

impl ScalarMul for Vec3 {
    fn scalar_mul(&self, s: f32) -> Self {
        let vec = Vec3::new_xyz(self.x() * s, self.y() * s, self.z() * s);
        return vec;
    }

    fn scalar_mul_(&mut self, s: f32) {
        self.data[0] *= s;
        self.data[1] *= s;
        self.data[2] *= s;
    }
}

impl Vec3 {
    pub fn from(v: &Vec4) -> Vec3 {
        Vec3::new_xyz(v.x(), v.y(), v.z())
    }

    pub(crate) fn _new() -> Self {
        Vec3::new(0.)
    }

    pub fn new(val: f32) -> Self {
        Vec3 {
            transposed: false,
            data: [val, val, val],
        }
    }

    pub fn new_xyz(x: f32, y: f32, z: f32) -> Self {
        Vec3 {
            transposed: false,
            data: [x, y, z],
        }
    }

    pub fn new_rgb(r: f32, g: f32, b: f32) -> Self {
        Vec3 {
            transposed: false,
            data: [r, g, b],
        }
    }

    #[inline]
    pub fn x(&self) -> f32 {
        self.data[0]
    }
    #[inline]
    pub fn y(&self) -> f32 {
        self.data[1]
    }
    #[inline]
    pub fn z(&self) -> f32 {
        self.data[2]
    }

    #[inline]
    pub fn r(&self) -> f32 {
        self.data[0]
    }
    #[inline]
    pub fn g(&self) -> f32 {
        self.data[1]
    }
    #[inline]
    pub fn b(&self) -> f32 {
        self.data[2]
    }

    #[inline]
    pub fn set_x(&mut self, x: f32) {
        self.data[0] = x;
    }
    #[inline]
    pub fn set_y(&mut self, y: f32) {
        self.data[1] = y;
    }
    #[inline]
    pub fn set_z(&mut self, z: f32) {
        self.data[2] = z;
    }

    #[inline]
    pub fn set_r(&mut self, r: f32) {
        self.data[0] = r;
    }
    #[inline]
    pub fn set_g(&mut self, g: f32) {
        self.data[1] = g;
    }
    #[inline]
    pub fn set_b(&mut self, b: f32) {
        self.data[2] = b;
    }
}

#[derive(Copy, Clone)]
pub struct Vec4 {
    transposed: bool,
    data: [f32; 4],
}

impl Product for Vec4 {
    fn product(&self, rhs: &Self) -> Self {
        let v = Vec4::new_xyzw(
            self.x() * rhs.x(),
            self.y() * rhs.y(),
            self.z() * rhs.z(),
            self.w() * rhs.w(),
        );
        return v;
    }

    fn product_(&mut self, rhs: &Self) {
        self.data[0] *= rhs.data[0];
        self.data[1] *= rhs.data[1];
        self.data[2] *= rhs.data[2];
        self.data[3] *= rhs.data[3];
    }
}

impl Vec for Vec4 {
    fn get(&self, index: usize) -> Result<f32, OutOfBoundError> {
        return if index > 3 {
            Err(OutOfBoundError::new([3, 0], [index, 0]))
        } else {
            Ok(self.data[index])
        };
    }

    fn set(&mut self, index: usize, val: f32) -> Result<(), OutOfBoundError> {
        return if index > 3 {
            Err(OutOfBoundError::new([3, 0], [index, 0]))
        } else {
            self.data[index] = val;
            Ok(())
        };
    }

    fn get_size(&self) -> usize {
        4
    }
}

impl VecDot for Vec4 {
    fn dot(&self, other: &Self) -> f32 {
        let accum = self.x() * other.x()
            + self.y() * other.y()
            + self.z() * other.z()
            + self.w() * other.w();
        return accum;
    }
}

impl Add for Vec4 {
    fn add(&self, other: &Self) -> Result<Self, DimensionMismatchError> {
        if self.transposed != other.transposed {
            return Err(DimensionMismatchError::new(
                if self.transposed { [1, 4] } else { [4, 1] },
                if other.transposed { [1, 4] } else { [4, 1] },
            ));
        } else {
            let d = [
                self.x() + other.x(),
                self.y() + other.y(),
                self.z() + other.z(),
                self.w() + other.w(),
            ];
            return Ok(Vec4 {
                data: d,
                transposed: self.transposed,
            });
        }
    }

    fn _add(&self, v: &Vec4) -> Vec4 {
        let data = [
            self.data[0] + v.data[0],
            self.data[1] + v.data[1],
            self.data[2] + v.data[2],
            self.data[3] + v.data[3],
        ];
        return Vec4 {
            transposed: false,
            data,
        };
    }

    fn add_(&mut self, other: &Self) {
        self.data[0] += other.data[0];
        self.data[1] += other.data[1];
        self.data[2] += other.data[2];
        self.data[3] += other.data[3];
    }
}

impl Minus for Vec4 {
    fn minus(&self, right: &Self) -> Result<Self, DimensionMismatchError> {
        if self.transposed != right.transposed {
            return Err(DimensionMismatchError::new(
                if self.transposed { [1, 4] } else { [4, 1] },
                if right.transposed { [1, 4] } else { [4, 1] },
            ));
        } else {
            let d = [
                self.x() - right.x(),
                self.y() - right.y(),
                self.z() - right.z(),
                self.w() - right.w(),
            ];
            return Ok(Vec4 {
                data: d,
                transposed: self.transposed,
            });
        }
    }

    fn minus_(&mut self, right: &Self) {
        self.data[0] -= right.data[0];
        self.data[1] -= right.data[1];
        self.data[2] -= right.data[2];
        self.data[3] -= right.data[3];
    }

    fn _minus(&self, right: &Self) -> Self {
        let data = [
            self.data[0] - right.data[0],
            self.data[1] - right.data[1],
            self.data[2] - right.data[2],
            self.data[3] - right.data[3],
        ];
        return Vec4 {
            transposed: false,
            data,
        };
    }
}

impl Transpose for Vec4 {
    fn transpose(&self) -> Self {
        let mut v = self.clone();
        v.transposed = !self.transposed;
        return v;
    }

    fn transpose_(&mut self) {
        self.transposed = !self.transposed;
    }
}

impl Length for Vec4 {
    fn get_length(&self) -> f32 {
        let x2 = self.data[0] * self.data[0];
        let y2 = self.data[1] * self.data[1];
        let z2 = self.data[2] * self.data[2];
        let w2 = self.data[3] * self.data[3];
        let l2 = x2 + y2 + z2 + w2;
        return l2.sqrt();
    }
}

impl Normalize for Vec4 {
    fn normalize(&self) -> Self {
        let l = self.get_length();
        Vec4::new_xyzw(
            self.data[0] / l,
            self.data[1] / l,
            self.data[2] / l,
            self.data[3] / l,
        )
    }

    fn normalize_(&mut self) {
        let l = self.get_length();
        self.data[0] /= l;
        self.data[1] /= l;
        self.data[2] /= l;
        self.data[3] /= l;
    }
}

impl ScalarMul for Vec4 {
    fn scalar_mul(&self, s: f32) -> Self {
        let vec = Vec4::new_xyzw(self.x() * s, self.y() * s, self.z() * s, self.w() * s);
        return vec;
    }

    fn scalar_mul_(&mut self, s: f32) {
        self.data[0] *= s;
        self.data[1] *= s;
        self.data[2] *= s;
        self.data[3] *= s;
    }
}

impl Vec4 {
    pub fn from(v: &Vec3, e4: f32) -> Vec4 {
        Vec4::new_xyzw(v.x(), v.y(), v.z(), e4)
    }

    pub(crate) fn _new() -> Self {
        Vec4::new(0.)
    }

    pub(crate) fn _set_all(&mut self, v: &Vec3, e4: f32) {
        self.data[0] = v.x();
        self.data[1] = v.y();
        self.data[2] = v.z();
        self.data[3] = e4;
    }

    pub fn new(val: f32) -> Self {
        Vec4 {
            transposed: false,
            data: [val, val, val, val],
        }
    }
    pub fn new_xyzw(x: f32, y: f32, z: f32, w: f32) -> Self {
        Vec4 {
            transposed: false,
            data: [x, y, z, w],
        }
    }

    pub fn new_rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        Vec4 {
            transposed: false,
            data: [r, g, b, a],
        }
    }

    #[inline]
    pub fn r(&self) -> f32 {
        self.data[0]
    }
    #[inline]
    pub fn g(&self) -> f32 {
        self.data[1]
    }
    #[inline]
    pub fn b(&self) -> f32 {
        self.data[2]
    }
    #[inline]
    pub fn a(&self) -> f32 {
        self.data[3]
    }

    #[inline]
    pub fn x(&self) -> f32 {
        self.data[0]
    }
    #[inline]
    pub fn y(&self) -> f32 {
        self.data[1]
    }
    #[inline]
    pub fn z(&self) -> f32 {
        self.data[2]
    }
    #[inline]
    pub fn w(&self) -> f32 {
        self.data[3]
    }

    #[inline]
    pub fn set_x(&mut self, x: f32) {
        self.data[0] = x;
    }
    #[inline]
    pub fn set_y(&mut self, y: f32) {
        self.data[1] = y;
    }
    #[inline]
    pub fn set_z(&mut self, z: f32) {
        self.data[2] = z;
    }

    #[inline]
    pub fn set_w(&mut self, w: f32) {
        self.data[3] = w;
    }

    #[inline]
    pub fn set_r(&mut self, r: f32) {
        self.data[0] = r;
    }
    #[inline]
    pub fn set_g(&mut self, g: f32) {
        self.data[1] = g;
    }
    #[inline]
    pub fn set_b(&mut self, b: f32) {
        self.data[2] = b;
    }

    #[inline]
    pub fn set_a(&mut self, a: f32) {
        self.data[3] = a;
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_cross() {
        let v1 = Vec3::new_xyz(1., 0., 0.);
        let v2 = Vec3::new_xyz(0., 1., 0.);
        let cross = v1.cross(&v2);
        assert_eq!(cross.z(), 1.);
        let v3 = Vec3::new_xyz(0., 0., 1.0);
        let cross = v1.cross(&v3);
        assert_eq!(cross.y(), -1.);
        let cross = v2.cross(&v3);
        assert_eq!(cross.x(), 1.);
    }
}
