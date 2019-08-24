use matrix::matrix::Matrix;
use matrix::matrixhandle::MatrixHandle;
use matrix::simplematrixhandle::SimpleMatrixHandle;
use std::ops::{ Add, AddAssign, Sub, SubAssign, Mul };
use serde::{ Deserializer, Deserialize, Serializer, Serialize };
use activationfunction::ActivationFunction;
use activationfunction::Sigmoid;
use activationfunction::Relu;
use activationfunction::TwoPlayerScore;
use matrix::MatrixHandleType;

pub struct VerifyingMatrixHandle {
    pub expected: SimpleMatrixHandle,
    pub actual: MatrixHandleType,
}

impl VerifyingMatrixHandle {
    pub fn verify(&self) {
        const epsilon: f32 = 0.0001;

        let expected_matrix = SimpleMatrixHandle::to_matrix(&self.expected);
        let actual_matrix = MatrixHandleType::to_matrix(&self.actual);

        if expected_matrix.rows() != actual_matrix.rows() {
            panic!("Different number of rows. Expected {} found {}", expected_matrix.rows(), actual_matrix.rows());
        }
        if expected_matrix.columns() != actual_matrix.columns() {
            panic!("Different number of columns. Expected {} found {}", expected_matrix.columns(), actual_matrix.columns());
        }

        let mut is_approx_equal = true;
        for i in 0..expected_matrix.rows() {
            for j in 0..expected_matrix.columns() {
                if (expected_matrix[(i, j)] - actual_matrix[(i, j)]).abs() > epsilon {
                    is_approx_equal = false;
                    break;
                }
            }
        }

        if !is_approx_equal {
            println!("Difference:\n{}", &expected_matrix - &actual_matrix);
            panic!("Failed to verify matrix handles!\nExpected:\n{}\n\nFound:\n{}", expected_matrix, actual_matrix);
        }
    }
}

impl MatrixHandle for VerifyingMatrixHandle {
    fn of_size(rows: usize, columns: usize) -> Self {
        return VerifyingMatrixHandle {
            expected: SimpleMatrixHandle::of_size(rows, columns),
            actual: MatrixHandleType::of_size(rows, columns)
        }
    }

    fn from_matrix(matrix: &Matrix<f32>) -> Self {
        return VerifyingMatrixHandle {
            expected: SimpleMatrixHandle::from_matrix(&matrix),
            actual: MatrixHandleType::from_matrix(&matrix)
        }
    }

    fn copy_from_matrix(dst: &mut Self, matrix: Matrix<f32>) {
        panic!("'copy_from_matrix' not implemented!");
    }

    fn to_matrix(&self) -> Matrix<f32> {
        self.verify();
        return self.expected.to_matrix();
    }

    fn copy(destination: &mut Self, source: &Self) {
        panic!("'copy' not implemented");
    }

    fn dropout_elements(&self, rate: f32) -> Self {
        panic!("'dropout_elements' not implemented");
    }

    fn dropout_rows(&self, rate: f32) -> Self {
        panic!("'dropout_rows' not implemented");
    }

    fn add_constant_row(&self, value: f32) -> Self {
        let res = VerifyingMatrixHandle {
            expected: self.expected.add_constant_row(value),
            actual: self.actual.add_constant_row(value)
        };
        res.verify();
        return res;
    }

    fn remove_first_row(&self) -> Self {
        let res = VerifyingMatrixHandle {
            expected: self.expected.remove_first_row(),
            actual: self.actual.remove_first_row()
        };
        res.verify();
        return res;
    }

    fn transpose(&self) -> Self {
        self.verify();
        let res = VerifyingMatrixHandle {
            expected: self.expected.transpose(),
            actual: self.actual.transpose()
        };
        res.verify();
        return res;
    }

    fn entrywise_product(&self, rhs: &VerifyingMatrixHandle) -> Self {
        let res = VerifyingMatrixHandle {
            expected: self.expected.entrywise_product(&rhs.expected),
            actual: self.actual.entrywise_product(&rhs.actual)
        };
        res.verify();
        return res;
    }

    fn inplace_entrywise_product(&mut self, rhs: &Self) {
        panic!("not implemented");
    }

    fn inplace_add_constant_row(&mut self, value: f32) {
        panic!("not implemented");
    }

    fn inplace_remove_first_row(&mut self) {
        panic!("not implemented");
    }

    fn inplace_transpose(&mut self) {
        panic!("not implemented");
    }

    fn inplace_scalar_multiply(&mut self, scalar: f32) {
        panic!("not implemented");
    }

    fn multiply(lhs: &Self, rhs: &Self, dst: &mut Self) {
        panic!("not implemented");
    }

    fn rows(&self) -> usize {
        self.verify();
        return self.expected.rows();
    }

    fn columns(&self) -> usize {
        self.verify();
        return self.expected.columns();
    }
}

impl Clone for VerifyingMatrixHandle {
    fn clone(&self) -> Self {
        let res = VerifyingMatrixHandle {
            expected: self.expected.clone(),
            actual: self.actual.clone()
        };
        res.verify();
        return res;
    }
}

impl Add for VerifyingMatrixHandle {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        return &self + &rhs;
    }
}
impl<'a> Add<&'a VerifyingMatrixHandle> for &'a VerifyingMatrixHandle {
    type Output = VerifyingMatrixHandle;

    fn add(self, rhs: &'a VerifyingMatrixHandle) -> VerifyingMatrixHandle {
        let res = VerifyingMatrixHandle {
            expected: &self.expected + &rhs.expected,
            actual: &self.actual + &rhs.actual
        };
        res.verify();
        return res;
    }
}

impl AddAssign for VerifyingMatrixHandle {
    fn add_assign(&mut self, rhs: Self) {
        self.expected += rhs.expected;
        self.actual += rhs.actual;
        self.verify();
    }
}

impl Sub for VerifyingMatrixHandle
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        return &self - &rhs;
    }
}
impl<'a> Sub<&'a VerifyingMatrixHandle> for &'a VerifyingMatrixHandle
{
    type Output = VerifyingMatrixHandle;

    fn sub(self, rhs: &'a VerifyingMatrixHandle) -> VerifyingMatrixHandle {
        let res = VerifyingMatrixHandle {
            expected: &self.expected - &rhs.expected,
            actual: &self.actual - &rhs.actual
        };
        res.verify();
        return res;
    }
}

impl SubAssign<VerifyingMatrixHandle> for VerifyingMatrixHandle {
    fn sub_assign(&mut self, rhs: Self) {
        panic!("not implemented");
    }
}
impl SubAssign<&VerifyingMatrixHandle> for VerifyingMatrixHandle {
    fn sub_assign(&mut self, rhs: &Self) {
        panic!("not implemented");
    }
}

impl Mul for VerifyingMatrixHandle {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        return &self * &rhs;
    }
}
impl<'a> Mul for &'a VerifyingMatrixHandle {
    type Output = VerifyingMatrixHandle;
    fn mul(self, rhs: &VerifyingMatrixHandle) -> VerifyingMatrixHandle {
        let res = VerifyingMatrixHandle {
            expected: &self.expected * &rhs.expected,
            actual: &self.actual * &rhs.actual
        };
        res.verify();
        return res;
    }
}

impl<'a> Mul<f32> for &'a VerifyingMatrixHandle {
    type Output = VerifyingMatrixHandle;
    fn mul(self, scalar: f32) -> VerifyingMatrixHandle {
        let res = VerifyingMatrixHandle {
            expected: &self.expected * scalar,
            actual: &self.actual * scalar
        };
        res.verify();
        return res;
    }
}
impl Mul<f32> for VerifyingMatrixHandle {
    type Output = VerifyingMatrixHandle;
    fn mul(self, scalar: f32) -> VerifyingMatrixHandle {
        return &self * scalar;
    }
}

impl Mul<VerifyingMatrixHandle> for f32 {
    type Output = VerifyingMatrixHandle;
    fn mul(self, scalar: VerifyingMatrixHandle) -> VerifyingMatrixHandle {
        return scalar * self;
    }
}
impl Mul<&VerifyingMatrixHandle> for f32 {
    type Output = VerifyingMatrixHandle;
    fn mul(self, scalar: &VerifyingMatrixHandle) -> VerifyingMatrixHandle {
        return scalar * self;
    }
}

impl Serialize for VerifyingMatrixHandle {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer
    {
        panic!("not implemented");
    }
}

impl<'de> Deserialize<'de> for VerifyingMatrixHandle {
    fn deserialize<D>(deserializer: D) -> Result<VerifyingMatrixHandle, D::Error>
        where D: Deserializer<'de>
    {
        panic!("not implemented");
    }
}

impl ActivationFunction<VerifyingMatrixHandle> for Sigmoid {
    fn evaluate(&self, input: &VerifyingMatrixHandle) -> VerifyingMatrixHandle {
        let res = VerifyingMatrixHandle {
            expected: self.evaluate(&input.expected),
            actual: self.evaluate(&input.actual)
        };
        res.verify();
        return res;
    }

    fn derivative(&self, input: &VerifyingMatrixHandle) -> VerifyingMatrixHandle {
        input.verify();
        let res = VerifyingMatrixHandle {
            expected: self.derivative(&input.expected),
            actual: self.derivative(&input.actual)
        };
        res.verify();
        return res;
    }

    fn inline_evaluate(&self, input: &mut VerifyingMatrixHandle) {
        panic!("Not implemented!");
    }

    fn inline_derivative(&self, input: &mut VerifyingMatrixHandle) {
        panic!("Not implemented!");
    }
}

impl ActivationFunction<VerifyingMatrixHandle> for Relu {
    fn evaluate(&self, input: &VerifyingMatrixHandle) -> VerifyingMatrixHandle {
        panic!("Not implemented!");
    }

    fn derivative(&self, input: &VerifyingMatrixHandle) -> VerifyingMatrixHandle {
        panic!("Not implemented!");
    }

    fn inline_evaluate(&self, input: &mut VerifyingMatrixHandle) {
        panic!("Not implemented!");
    }

    fn inline_derivative(&self, input: &mut VerifyingMatrixHandle) {
        panic!("Not implemented!");
    }
}

impl ActivationFunction<VerifyingMatrixHandle> for TwoPlayerScore {
    fn evaluate(&self, input: &VerifyingMatrixHandle) -> VerifyingMatrixHandle {
        panic!("Not implemented!");
    }

    fn derivative(&self, input: &VerifyingMatrixHandle) -> VerifyingMatrixHandle {
        panic!("Not implemented!");
    }

    fn inline_evaluate(&self, input: &mut VerifyingMatrixHandle) {
        panic!("Not implemented!");
    }

    fn inline_derivative(&self, input: &mut VerifyingMatrixHandle) {
        panic!("Not implemented!");
    }
}
