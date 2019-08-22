use matrixhandle::MatrixHandle;
use matrix::Matrix;
use std::ops::{ Add, AddAssign, Sub, SubAssign, Mul, MulAssign };
use serde::{ Deserializer, Deserialize, Serializer, Serialize };

pub struct SimpleMatrixHandle {
    matrix: Matrix<f32>
}

impl MatrixHandle for SimpleMatrixHandle {
    fn of_size(rows: usize, columns: usize) -> SimpleMatrixHandle {
        // Just some arbitrary matrix, as we will allocate an entire matrix anyways
        return Self::from_matrix(&Matrix::new(1, 1, &|row, col| 0_f32));
    }

    fn from_matrix(matrix: &Matrix<f32>) -> SimpleMatrixHandle {
        return SimpleMatrixHandle { matrix: matrix.clone() }
    }

    fn copy_from_matrix(dst: &mut Self, matrix: Matrix<f32>) {
        dst.matrix = matrix.clone();
    }

    fn to_matrix(&self) -> Matrix<f32> {
        return self.matrix.clone();
    }

    fn copy(destination: &mut Self, source: &Self) {
        destination.matrix = source.matrix.clone();
    }

    fn dropout_elements(&self, rate: f32) -> Self {
        return Self::from_matrix(&self.matrix.dropout_elements(rate as f64));
    }

    fn dropout_rows(&self, rate: f32) -> Self {
        return Self::from_matrix(&self.matrix.dropout_rows(rate as f64));
    }

    fn add_constant_row(&self, value: f32) -> Self {
        return Self::from_matrix(&self.matrix.add_constant_row(value));
    }

    fn remove_first_row(&self) -> Self {
        return Self::from_matrix(&self.matrix.remove_first_row());
    }

    fn transpose(&self) -> Self {
        return Self::from_matrix(&self.matrix.transpose());
    }

    fn entrywise_product(&self, rhs: &SimpleMatrixHandle) -> Self {
        return Self::from_matrix(&self.matrix.entrywise_product(&rhs.matrix));
    }

    fn inplace_entrywise_product(&mut self, rhs: &Self) {
        self.matrix = self.matrix.entrywise_product(&rhs.matrix);
    }

    fn inplace_add_constant_row(&mut self, value: f32) {
        self.matrix = self.matrix.add_constant_row(value);
    }

    fn inplace_remove_first_row(&mut self) {
        self.matrix = self.matrix.remove_first_row();
    }

    fn inplace_transpose(&mut self) {
        self.matrix = self.matrix.transpose();
    }

    fn inplace_scalar_multiply(&mut self, scalar: f32) {
        self.matrix = scalar * &self.matrix;
    }

    fn multiply(lhs: &Self, rhs: &Self, dst: &mut Self) {
        dst.matrix = &lhs.matrix * &rhs.matrix;
    }

    fn rows(&self) -> usize {
        return self.matrix.rows();
    }

    fn columns(&self) -> usize {
        return self.matrix.columns();
    }
}

impl Clone for SimpleMatrixHandle {
    fn clone(&self) -> SimpleMatrixHandle {
        return SimpleMatrixHandle::from_matrix(&self.matrix.clone());
    }
}

impl Add for SimpleMatrixHandle {
    type Output = SimpleMatrixHandle;
    fn add(self, rhs: SimpleMatrixHandle) -> SimpleMatrixHandle {
        return &self + &rhs;
    }
}
impl<'a> Add<&'a SimpleMatrixHandle> for &'a SimpleMatrixHandle {
    type Output = SimpleMatrixHandle;

    fn add(self, rhs: &'a SimpleMatrixHandle) -> SimpleMatrixHandle {
        return SimpleMatrixHandle::from_matrix(&(&self.matrix + &rhs.matrix));
    }
}

impl AddAssign for SimpleMatrixHandle {
    fn add_assign(&mut self, rhs: SimpleMatrixHandle) {
        self.matrix += &rhs.matrix;
    }
}
impl<'a> AddAssign<&'a SimpleMatrixHandle> for SimpleMatrixHandle {
    fn add_assign(&mut self, rhs: &'a SimpleMatrixHandle) {
        self.matrix += &rhs.matrix;
    }
}

impl Sub for SimpleMatrixHandle {
    type Output = SimpleMatrixHandle;

    fn sub(self, rhs: SimpleMatrixHandle) -> SimpleMatrixHandle {
        return &self - &rhs;
    }
}
impl<'a> Sub<&'a SimpleMatrixHandle> for &'a SimpleMatrixHandle {
    type Output = SimpleMatrixHandle;

    fn sub(self, rhs: &'a SimpleMatrixHandle) -> SimpleMatrixHandle {
        return SimpleMatrixHandle::from_matrix(&(&self.matrix - &rhs.matrix));
    }
}

impl SubAssign<SimpleMatrixHandle> for SimpleMatrixHandle {
    fn sub_assign(&mut self, rhs: SimpleMatrixHandle) {
        self.matrix -= &rhs.matrix;
    }
}
impl SubAssign<&SimpleMatrixHandle> for SimpleMatrixHandle {
    fn sub_assign(&mut self, rhs: &SimpleMatrixHandle) {
        self.matrix -= &rhs.matrix;
    }
}

impl Mul for SimpleMatrixHandle {
    type Output = SimpleMatrixHandle;
    fn mul(self, rhs: SimpleMatrixHandle) -> SimpleMatrixHandle {
        return &self * &rhs;
    }
}
impl<'a> Mul for &'a SimpleMatrixHandle {
    type Output = SimpleMatrixHandle;
    fn mul(self, rhs: &SimpleMatrixHandle) -> SimpleMatrixHandle {
        return SimpleMatrixHandle::from_matrix(&(&self.matrix * &rhs.matrix));
    }
}

impl<'a> Mul<f32> for &'a SimpleMatrixHandle {
    type Output = SimpleMatrixHandle;
    fn mul(self, scalar: f32) -> SimpleMatrixHandle {
        return SimpleMatrixHandle::from_matrix(&(scalar * &self.matrix));
    }
}
impl Mul<f32> for SimpleMatrixHandle {
    type Output = SimpleMatrixHandle;
    fn mul(self, scalar: f32) -> SimpleMatrixHandle {
        return &self * scalar;
    }
}

impl Mul<SimpleMatrixHandle> for f32 {
    type Output = SimpleMatrixHandle;
    fn mul(self, scalar: SimpleMatrixHandle) -> SimpleMatrixHandle {
        return scalar * self;
    }
}
impl Mul<&SimpleMatrixHandle> for f32 {
    type Output = SimpleMatrixHandle;
    fn mul(self, scalar: &SimpleMatrixHandle) -> SimpleMatrixHandle {
        return scalar * self;
    }
}

impl Serialize for SimpleMatrixHandle {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer
    {
        self.matrix.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SimpleMatrixHandle {
    fn deserialize<D>(deserializer: D) -> Result<SimpleMatrixHandle, D::Error>
        where D: Deserializer<'de>
    {
        let deserialize_result = Matrix::deserialize(deserializer);
        deserialize_result.map(|matrix| SimpleMatrixHandle::from_matrix(&matrix))
    }
}
