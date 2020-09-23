use std::cell::RefCell;
use crate::err::{DimensionMismatchError, OutOfBoundError};

pub trait VecDot<Rhs = Self>
{
    fn dot(&self, rhs: &Rhs) -> f32;
}

pub trait MatVecDot<Rhs: Vec>
{
    fn mat_vec_dot(&self, rhs: &Rhs) -> Rhs;
}

pub trait Product<Rhs = Self>
{
    fn product(&self, rhs: &Rhs) -> Rhs;
    fn product_(&mut self, rhs: &Rhs);
}

pub trait Mat
{
    fn get_entry(&self, row: usize, col: usize) -> Result<f32, OutOfBoundError>;
    fn set_entry(&mut self, row: usize, col: usize, val: f32) -> Result<(), OutOfBoundError>;
    fn get_size(&self) -> [usize; 2];
}

pub(crate) trait _Mat: Mat
{
    fn _get_entry(&self, row: usize, col: usize) -> f32
    {
        self.get_entry(row, col).expect("Should NOT happen")
    }
    fn _set_entry(&mut self, row: usize, col: usize, val: f32)
    {
        self.set_entry(row, col, val).expect("Should NOT happen")
    }
}

// set default implementation for all Mat
impl<T: Mat> _Mat for T {}

pub(crate) trait _Vec: Vec
{
    fn _get(&self, index: usize) -> f32
    {
        self.get(index).expect("Should NOT happen")
    }

    fn _set(&mut self, index: usize, val: f32)
    {
        self.set(index, val).expect("Should NOT happen")
    }
}

// set default implementation for all Vec
impl<T: Vec> _Vec for T {}

pub trait Vec
{
    fn get(&self, index: usize) -> Result<f32, OutOfBoundError>;
    fn set(&mut self, index: usize, val: f32) -> Result<(), OutOfBoundError>;
    fn get_size(&self) -> usize;
}

pub trait ScalarMul<Output = Self>
{
    fn scalar_mul(&self, s: f32) -> Output;
    fn scalar_mul_(&mut self, s: f32);
}

pub trait Cross<Rhs = Self>
{
    fn cross(&self, other: &Rhs) -> Rhs;
}

pub trait Add<Output = Self>
{
    fn add(&self, other: &Output) -> Result<Output, DimensionMismatchError>;
    fn add_(&mut self, other: &Output);
    fn _add(&self, other: &Output) -> Output;
}

pub trait Minus<Rhs = Self>
{
    fn minus(&self, right: &Rhs) -> Result<Rhs, DimensionMismatchError>;
    fn _minus(&self, right: &Rhs) -> Rhs;
}

pub trait Transpose<Output = Self>
{
    fn transpose(&self) -> Output;
    fn transpose_(&mut self);
}

/// A trait enabling vector normalization
/// # Notice
/// The implementation should not worry about zero vector
pub trait Normalize<Output = Self>
{
    fn normalize(&self) -> Output;
    fn normalize_(&mut self);
}

pub trait Length
{
    fn get_length(&self) -> f32;
}

/// A trait enabling matrix inverse
/// # Notice
/// The implementation should not worry about zero vector
pub trait Inverse<Output = Self>
{
    fn inverse(&self) -> Output;
}

// column first storage
#[derive(Clone)]
pub struct Mat4
{
    pub(self) transposed: bool,
    pub(self) data: [f32; 16],
    inverse: RefCell<[f32; 16]>,
    inverted: RefCell<bool>
}

impl Mat4 {
    pub fn identity() -> Self
    {
        let mut data = [0.; 16];
        for i in 0..4
        {
            data[i * 4 + i] = 1.;
        }
        Mat4
        {
            transposed: false,
            data,
            inverse: RefCell::new([0.; 16]),
            inverted: RefCell::new(false)
        }
    }

    pub(crate) fn _new1(transposed: bool, data: [f32; 16]) -> Self
    {
        Mat4 {
            transposed,
            data,
            inverse: RefCell::new([0.0; 16]),
            inverted: RefCell::new(false)
        }
    }
    pub(crate) fn _new2(transposed: bool, data: [f32; 16], inverse: RefCell<[f32; 16]>, inverted: RefCell<bool>) -> Self
    {
        Mat4 {
            transposed,
            data,
            inverse,
            inverted
        }
    }

    pub(crate) fn _set_row(&mut self, row: usize, val: &Vec4)
    {
        for i in 0..4 {
            self._set_entry(row, i, val._get(i));
        }
    }

    pub(crate) fn _get_row(&self, row: usize) -> Vec4
    {
        let mut v = Vec4::_new();
        for i in 0..4
        {
            v._set(i, self._get_entry(row, i));
        }
        return v;
    }

    pub(crate) fn _set_column(&mut self, column: usize, val: &Vec4)
    {
        for i in 0..4 {
            self._set_entry(i, column, val._get(i));
        }
    }

    pub(crate) fn _get_column(&self, column: usize) -> Vec4
    {
        let mut v = Vec4::_new();
        for i in 0..4
        {
            v._set(i, self._get_entry(i, column));
        }
        return v;
    }

    #[inline]
    fn transposed_get(&self, row: usize, col: usize) -> f32
    {
        self.data[col * 4 + row]
    }
    #[inline]
    fn get(&self, row: usize, col: usize) -> f32
    {
        self.data[row * 4 + col]
    }

    #[inline]
    fn clear_inverse(&mut self)
    {
        if *self.inverted.borrow()
        {
            self.inverted = RefCell::new(false);
        }
    }

    pub fn dot_mat(&self, other: &Mat4) -> Mat4
    {
        let mut prod = [0.0; 16];
        let self_get = if self.transposed { Mat4::transposed_get } else { Mat4::get };
        let other_get = if other.transposed { Mat4::transposed_get } else { Mat4::get };
        let mut entry;
        for row in 0..4 {
            for col in 0..4 {
                entry = 0.0;
                for idx in 0..4 {
                    entry += self_get(self, row, idx) * other_get(other, idx, col);
                }
                prod[4 * row + col] = entry;
            }
        }
        return Mat4 {
            transposed: false,
            data: prod,
            inverse: RefCell::new([0.0; 16]),
            inverted: RefCell::new(false)
        }
    }
}

impl MatVecDot<Vec4> for Mat4
{
    fn mat_vec_dot(&self, rhs: &Vec4) -> Vec4 {
        let mut v = Vec4::_new();
        for row in 0..4
        {
            let mut accum = 0.;
            for col in 0..4
            {
                accum += self._get_entry(row, col) * rhs._get(col);
            }
            v._set(row, accum);
        }
        return v;
    }
}

impl Mat for Mat4 {
    fn get_entry(&self, row: usize, col: usize) -> Result<f32, OutOfBoundError> {
        return if row > 3 || col > 3
        {
            Err(OutOfBoundError::new([3, 3], [row, col]))
        } else {
            Ok(if self.transposed { self.transposed_get(row, col) } else { self.get(row, col) })
        }
    }

    fn set_entry(&mut self, row: usize, col: usize, val: f32) -> Result<(), OutOfBoundError> {
        return if row >= 4 || col >= 4
        {
            Err(OutOfBoundError::new([3, 3], [row, col]))
        } else {
            self.clear_inverse();
            if self.transposed
            {
                self.data[col * 4 + row] = val;
            } else {
                self.data[row * 4 + col] = val;
            }
            Ok(())
        }
    }

    fn get_size(&self) -> [usize; 2]
    {
        [4, 4]
    }
}

impl ScalarMul for Mat4
{
    fn scalar_mul(&self, s: f32) -> Self {
        let mut data = [0.0; 16];
        for i in 0..16
        {
            data[i] = self.data[i] * s;
        }
        Mat4 {
            transposed: self.transposed,
            data,
            inverse: RefCell::new([0.0; 16]), // optimization needed here
            inverted: RefCell::new(false)
        }
    }

    fn scalar_mul_(&mut self, s: f32) {
        for i in 0..16
        {
            self.data[i] *= s;
        }
        self.inverted = RefCell::new(false);
        self.inverse = RefCell::new([0.0; 16]);
    }
}

impl Inverse for Mat4
{
    fn inverse(&self) -> Self {
        unimplemented!()
    }
}

impl Transpose for Mat4
{
    fn transpose(&self) -> Self {
        let mut t_data: [f32; 16] = [0.0; 16];
        if !self.transposed
        {
            for i in 0..4 {
                for j in 0..4 {
                    t_data[j * 4 + i] = self.data[i * 4 + j];
                }
            }
        } else {
            t_data = self.data;
        }
        return Mat4
        {
            transposed: false,
            data: t_data,
            inverse: RefCell::new([0.0; 16]), // optimize later
            inverted: RefCell::new(false)
        }
    }

    fn transpose_(&mut self) {
        self.transposed = !self.transposed;
        self.inverted = RefCell::new(false);
    }
}

// column first storage
#[derive(Clone)]
pub struct Mat3
{
    transposed: bool,
    data: [f32; 9],
    inverse: RefCell<[f32; 9]>,
    inverted: RefCell<bool>
}

impl Mat3 {
    pub fn identity() -> Self
    {
        let mut data = [0.; 9];
        for i in 0..3
        {
            data[i * 3 + i] = 1.;
        }
        Mat3
        {
            transposed: false,
            data,
            inverse: RefCell::new([0.; 9]),
            inverted: RefCell::new(false)
        }
    }
    pub(crate) fn _new1(transposed: bool, data: [f32; 9]) -> Self
    {
        Mat3 {
            transposed,
            data,
            inverse: RefCell::new([0.0; 9]),
            inverted: RefCell::new(false)
        }
    }
    pub(crate) fn _new2(transposed: bool, data: [f32; 9], inverse: RefCell<[f32; 9]>, inverted: RefCell<bool>) -> Self
    {
        Mat3 {
            transposed,
            data,
            inverse,
            inverted
        }
    }

    pub(crate) fn _set_row(&mut self, row: usize, val: &Vec3)
    {
        for i in 0..3 {
            self._set_entry(row, i, val._get(i));
        }
    }

    pub(crate) fn _get_row(&self, row: usize) -> Vec3
    {
        let mut v = Vec3::_new();
        for i in 0..3
        {
            v._set(i, self._get_entry(row, i));
        }
        return v;
    }

    pub(crate) fn _set_column(&mut self, column: usize, val: &Vec3)
    {
        for i in 0..3 {
            self._set_entry(i, column, val._get(i));
        }
    }

    pub(crate) fn _get_column(&self, column: usize) -> Vec3
    {
        let mut v = Vec3::_new();
        for i in 0..3
        {
            v._set(i, self._get_entry(i, column));
        }
        return v;
    }

    #[inline]
    fn transposed_get(&self, row: usize, col: usize) -> f32
    {
        self.data[col * 3 + row]
    }

    #[inline]
    fn clear_inverse(&mut self)
    {
        if *self.inverted.borrow()
        {
            self.inverted = RefCell::new(false);
        }
    }

    #[inline]
    fn get(&self, row: usize, col: usize) -> f32
    {
        self.data[row * 3 + col]
    }

    pub fn dot_mat(&self, other: &Mat3) -> Mat3
    {
        let mut prod = [0.0; 9];
        let self_get = if self.transposed { Mat3::transposed_get } else { Mat3::get };
        let other_get = if other.transposed { Mat3::transposed_get } else { Mat3::get };
        let mut entry;
        for row in 0..3 {
            for col in 0..3 {
                entry = 0.0;
                for idx in 0..3 {
                    entry += self_get(self, row, idx) * other_get(other, idx, col);
                }
                prod[3 * row + col] = entry;
            }
        }
        return Mat3 {
            transposed: false,
            data: prod,
            inverse: RefCell::new([0.0; 9]),
            inverted: RefCell::new(false)
        }
    }
}

impl MatVecDot<Vec3> for Mat3
{
    fn mat_vec_dot(&self, rhs: &Vec3) -> Vec3 {
        let mut v = Vec3::_new();
        for row in 0..3
        {
            let mut accum = 0.;
            for col in 0..3
            {
                accum += self._get_entry(row, col) * rhs._get(col);
            }
            v._set(row, accum);
        }
        return v;
    }
}

impl Mat for Mat3 {
    fn get_entry(&self, row: usize, col: usize) -> Result<f32, OutOfBoundError> {
        return if row > 2 || col > 2
        {
            Err(OutOfBoundError::new([2, 2], [row, col]))
        } else {
            Ok(if self.transposed { self.transposed_get(row, col) } else { self.get(row, col) })
        }
    }

    fn set_entry(&mut self, row: usize, col: usize, val: f32) -> Result<(), OutOfBoundError> {
        return if row >= 3 || col >= 3
        {
            Err(OutOfBoundError::new([2, 2], [row, col]))
        } else {
            self.clear_inverse();
            if self.transposed
            {
                self.data[col * 3 + row] = val;
            } else {
                self.data[row * 3 + col] = val;
            }
            Ok(())
        }
    }

    fn get_size(&self) -> [usize; 2] {
        [3, 3]
    }
}

impl ScalarMul for Mat3
{
    fn scalar_mul(&self, s: f32) -> Self {
        let mut data = [0.0; 9];
        for i in 0..9
        {
            data[i] = self.data[i] * s;
        }
        Mat3 {
            transposed: self.transposed,
            data,
            inverse: RefCell::new([0.0; 9]), // optimization needed here
            inverted: RefCell::new(false)
        }
    }

    fn scalar_mul_(&mut self, s: f32) {
        for i in 0..9
        {
            self.data[i] *= s;
        }
        self.inverted = RefCell::new(false);
        self.inverse = RefCell::new([0.0; 9]);
    }
}

impl Transpose for Mat3
{
    fn transpose(&self) -> Self {
        let mut t_data: [f32; 9] = [0.0; 9];
        if !self.transposed
        {
            for i in 0..3 {
                for j in 0..3 {
                    t_data[j * 3 + i] = self.data[i * 3 + j];
                }
            }
        } else {
            t_data = self.data;
        }
        return Mat3 {
            transposed: false,
            data: t_data,
            inverse: RefCell::new([0.0; 9]),
            inverted: RefCell::new(false)
        }
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

#[derive(Copy, Clone)]
pub struct Vec3
{
    transposed: bool,
    data: [f32; 3]
}

impl Vec for Vec3
{
    fn get(&self, index: usize) -> Result<f32, OutOfBoundError> {
        return if index > 2 { Err(OutOfBoundError::new([2, 0], [index, 0])) } else { Ok(self.data[index]) }
    }

    fn set(&mut self, index: usize, val: f32) -> Result<(), OutOfBoundError> {
        return if index > 2
        {
            Err(OutOfBoundError::new([2, 0], [index, 0]))
        } else {
            self.data[index] = val;
            Ok(())
        }
    }

    fn get_size(&self) -> usize {
        3
    }
}

impl VecDot for Vec3 {
    fn dot(&self, other: &Self) -> f32 {
        let mut accum = 0.0;
        for i in 0..3 {
            accum += self.data[i] * other.data[i];
        }
        return accum;
    }
}

impl Add for Vec3
{
    fn add(&self, other: &Vec3) -> Result<Self, DimensionMismatchError> {
        if self.transposed != other.transposed {
            return Err(DimensionMismatchError::new(
                if self.transposed { [1, 3] } else { [3, 1] },
                if other.transposed { [1, 3] } else { [3, 1] }
            ))
        } else {
            let mut d = [0.0; 3];
            for i in 0..3 {
                d[i] = self.data[i] + other.data[i];
            }
            return Ok(Vec3 {
                data: d,
                transposed: self.transposed
            })
        }
    }

    fn add_(&mut self, other: &Self) {
        for i in 0..3
        {
            self.data[i] += other.data[i];
        }
    }

    fn _add(&self, v: &Vec3) -> Vec3
    {
        let mut data = [self.data[0] + v.data[0],
            self.data[1] + v.data[1],
            self.data[2] + v.data[2],
        ];
        return Vec3
        {
            transposed: false,
            data
        }
    }
}

impl Minus for Vec3 {
    fn minus(&self, right: &Self) -> Result<Self, DimensionMismatchError> {
        if self.transposed != right.transposed {
            return Err(DimensionMismatchError::new(
                if self.transposed { [1, 3] } else { [3, 1] },
                if right.transposed { [1, 3] } else { [3, 1] }
            ))
        } else {
            let mut d = [0.0; 3];
            for i in 0..3 {
                d[i] = self.data[i] - right.data[i];
            }
            return Ok(Vec3 {
                data: d,
                transposed: self.transposed
            })
        }
    }

    fn _minus(&self, right: &Self) -> Self {
        let mut data = [self.data[0] - right.data[0],
            self.data[1] - right.data[1],
            self.data[2] - right.data[2],
        ];
        return Vec3
        {
            transposed: false,
            data
        }
    }
}

impl Cross for Vec3
{
    fn cross(&self, right: &Self) -> Self {
        let d = [self.y() * right.z() - self.z() * right.y(),
            self.z() * right.x() - self.x() * right.z(),
            self.x() * right.y() - self.y() * right.x()];
        Vec3 {
            data: d,
            transposed: false
        }
    }
}

impl Transpose for Vec3
{
    fn transpose(&self) -> Self {
        let mut v = self.clone();
        v.transposed = !self.transposed;
        return v;
    }

    fn transpose_(&mut self) {
        self.transposed = !self.transposed;
    }
}

impl Length for Vec3
{
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
        let mut v = self.clone();
        for i in 0..3 {
            v.data[i] *= rhs.data[i]
        }
        return v;
    }

    fn product_(&mut self, rhs: &Self) {
        for i in 0..3
        {
            self.data[i] *= rhs.data[i];
        }
    }
}

impl Normalize for Vec3
{
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

impl ScalarMul for Vec3
{
    fn scalar_mul(&self, s: f32) -> Self {
        let mut vec = self.clone();
        for i in 0..3 {
            vec.data[i] *= s;
        }
        return vec;
    }

    fn scalar_mul_(&mut self, s: f32) {
        for i in 0..3 {
            self.data[i] *= s;
        }
    }
}


impl Vec3
{
    pub fn from(v: &Vec4) -> Vec3
    {
        Vec3::new_xyz(v.x(), v.y(), v.z())
    }

    pub(crate) fn _new() -> Self
    {
        Vec3::new(0.)
    }

    pub fn new(val: f32) -> Self
    {
        Vec3
        {
            transposed: false,
            data: [val, val, val]
        }
    }

    pub fn new_xyz(x: f32, y: f32, z: f32) -> Self
    {
        Vec3
        {
            transposed: false,
            data: [x, y, z]
        }
    }

    pub fn new_rgb(r: f32, g: f32, b: f32) -> Self
    {
        Vec3
        {
            transposed: false,
            data: [r, g, b]
        }
    }

    #[inline]
    pub fn x(&self) -> f32
    {
        self.data[0]
    }
    #[inline]
    pub fn y(&self) -> f32
    {
        self.data[1]
    }
    #[inline]
    pub fn z(&self) -> f32
    {
        self.data[2]
    }

    #[inline]
    pub fn r(&self) -> f32
    {
        self.data[0]
    }
    #[inline]
    pub fn g(&self) -> f32
    {
        self.data[1]
    }
    #[inline]
    pub fn b(&self) -> f32
    {
        self.data[2]
    }

    #[inline]
    pub fn set_x(&mut self, x: f32)
    {
        self.data[0] = x;
    }
    #[inline]
    pub fn set_y(&mut self, y: f32)
    {
        self.data[1] = y;
    }
    #[inline]
    pub fn set_z(&mut self, z: f32)
    {
        self.data[2] = z;
    }

    #[inline]
    pub fn set_r(&mut self, r: f32)
    {
        self.data[0] = r;
    }
    #[inline]
    pub fn set_g(&mut self, g: f32)
    {
        self.data[1] = g;
    }
    #[inline]
    pub fn set_b(&mut self, b: f32)
    {
        self.data[2] = b;
    }
}

#[derive(Copy, Clone)]
pub struct Vec4
{
    transposed: bool,
    data: [f32; 4]
}

impl Product for Vec4
{
    fn product(&self, rhs: &Self) -> Self {
        let mut v = self.clone();
        for i in 0..4 {
            v.data[i] *= rhs.data[i]
        }
        return v;
    }

    fn product_(&mut self, rhs: &Self) {
        for i in 0..4
        {
            self.data[i] *= rhs.data[i];
        }
    }
}

impl Vec for Vec4
{
    fn get(&self, index: usize) -> Result<f32, OutOfBoundError> {
        return if index > 3 { Err(OutOfBoundError::new([3, 0], [index, 0])) } else { Ok(self.data[index]) }
    }

    fn set(&mut self, index: usize, val: f32) -> Result<(), OutOfBoundError> {
        return if index > 3
        {
            Err(OutOfBoundError::new([3, 0], [index, 0]))
        } else {
            self.data[index] = val;
            Ok(())
        }
    }

    fn get_size(&self) -> usize {
        4
    }
}

impl VecDot for Vec4 {
    fn dot(&self, other: &Self) -> f32 {
        let mut accum = 0.0;
        for i in 0..4 {
            accum += self.data[i] * other.data[i];
        }
        return accum;
    }
}

impl Add for Vec4
{
    fn add(&self, other: &Self) -> Result<Self, DimensionMismatchError> {
        if self.transposed != other.transposed {
            return Err(DimensionMismatchError::new(
                if self.transposed { [1, 4] } else { [4, 1] },
                if other.transposed { [1, 4] } else { [4, 1] }
            ))
        } else {
            let mut d = [0.0; 4];
            for i in 0..4 {
                d[i] = self.data[i] + other.data[i];
            }
            return Ok(Vec4 {
                data: d,
                transposed: self.transposed
            })
        }
    }

    fn _add(&self, v: &Vec4) -> Vec4
    {
        let mut data = [self.data[0] + v.data[0],
            self.data[1] + v.data[1],
            self.data[2] + v.data[2],
            self.data[3] + v.data[3]];
        return Vec4
        {
            transposed: false,
            data
        }
    }

    fn add_(&mut self, other: &Self) {
        for i in 0..4
        {
            self.data[i] += other.data[i];
        }
    }
}

impl Minus for Vec4 {
    fn minus(&self, right: &Self) -> Result<Self, DimensionMismatchError> {
        if self.transposed != right.transposed {
            return Err(DimensionMismatchError::new(
                if self.transposed { [1, 4] } else { [4, 1] },
                if right.transposed { [1, 4] } else { [4, 1] }
            ))
        } else {
            let mut d = [0.0; 4];
            for i in 0..4 {
                d[i] = self.data[i] - right.data[i];
            }
            return Ok(Vec4 {
                data: d,
                transposed: self.transposed
            })
        }
    }
    fn _minus(&self, right: &Self) -> Self {
        let mut data = [self.data[0] - right.data[0],
            self.data[1] - right.data[1],
            self.data[2] - right.data[2],
            self.data[3] - right.data[3]
        ];
        return Vec4
        {
            transposed: false,
            data
        }
    }
}

impl Transpose for Vec4
{
    fn transpose(&self) -> Self {
        let mut v = self.clone();
        v.transposed = !self.transposed;
        return v;
    }

    fn transpose_(&mut self) {
        self.transposed = !self.transposed;
    }
}

impl Length for Vec4
{
    fn get_length(&self) -> f32 {
        let x2 = self.data[0] * self.data[0];
        let y2 = self.data[1] * self.data[1];
        let z2 = self.data[2] * self.data[2];
        let w2 = self.data[3] * self.data[3];
        let l2 = x2 + y2 + z2 + w2;
        return l2.sqrt();
    }
}

impl Normalize for Vec4
{
    fn normalize(&self) -> Self {
        let l = self.get_length();
        Vec4::new_xyzw(self.data[0] / l, self.data[1] / l, self.data[2] / l, self.data[3] / l)
    }

    fn normalize_(&mut self) {
        let l = self.get_length();
        self.data[0] /= l;
        self.data[1] /= l;
        self.data[2] /= l;
        self.data[3] /= l;
    }
}

impl ScalarMul for Vec4
{
    fn scalar_mul(&self, s: f32) -> Self {
        let mut vec = self.clone();
        for i in 0..4 {
            vec.data[i] *= s;
        }
        return vec;
    }

    fn scalar_mul_(&mut self, s: f32) {
        for i in 0..4 {
            self.data[i] *= s;
        }
    }
}

impl Vec4 {
    pub fn from(v: &Vec3, e4: f32) -> Vec4
    {
        Vec4::new_xyzw(v.x(), v.y(), v.z(), e4)
    }

    pub(crate) fn _new() -> Self
    {
        Vec4::new(0.)
    }

    pub(crate) fn _set_all(&mut self, v: &Vec3, e4: f32)
    {
        self.data[0] = v.x();
        self.data[1] = v.y();
        self.data[2] = v.z();
        self.data[3] = e4;
    }

    pub fn new(val: f32) -> Self
    {
        Vec4
        {
            transposed: false,
            data: [val, val, val, val]
        }
    }
    pub fn new_xyzw(x: f32, y: f32, z: f32, w: f32) -> Self
    {
        Vec4
        {
            transposed: false,
            data: [x, y, z, w]
        }
    }

    pub fn new_rgba(r: f32, g: f32, b: f32, a: f32) -> Self
    {
        Vec4
        {
            transposed: false,
            data: [r, g, b, a]
        }
    }

    #[inline]
    pub fn r(&self) -> f32
    {
        self.data[0]
    }
    #[inline]
    pub fn g(&self) -> f32
    {
        self.data[1]
    }
    #[inline]
    pub fn b(&self) -> f32
    {
        self.data[2]
    }
    #[inline]
    pub fn a(&self) -> f32
    {
        self.data[3]
    }

    #[inline]
    pub fn x(&self) -> f32
    {
        self.data[0]
    }
    #[inline]
    pub fn y(&self) -> f32
    {
        self.data[1]
    }
    #[inline]
    pub fn z(&self) -> f32
    {
        self.data[2]
    }
    #[inline]
    pub fn w(&self) -> f32
    {
        self.data[3]
    }

    #[inline]
    pub fn set_x(&mut self, x: f32)
    {
        self.data[0] = x;
    }
    #[inline]
    pub fn set_y(&mut self, y: f32)
    {
        self.data[1] = y;
    }
    #[inline]
    pub fn set_z(&mut self, z: f32)
    {
        self.data[2] = z;
    }

    #[inline]
    pub fn set_w(&mut self, w: f32)
    {
        self.data[3] = w;
    }

    #[inline]
    pub fn set_r(&mut self, r: f32)
    {
        self.data[0] = r;
    }
    #[inline]
    pub fn set_g(&mut self, g: f32)
    {
        self.data[1] = g;
    }
    #[inline]
    pub fn set_b(&mut self, b: f32)
    {
        self.data[2] = b;
    }

    #[inline]
    pub fn set_a(&mut self, a: f32)
    {
        self.data[3] = a;
    }
}


#[cfg(test)]
mod test
{
    use super::*;

    #[test]
    fn test_cross()
    {
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