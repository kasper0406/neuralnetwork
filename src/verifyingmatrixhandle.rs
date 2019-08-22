use matrix::Matrix;
use matrixhandle::MatrixHandle;
use simplematrixhandle::SimpleMatrixHandle;
use metalmatrixhandle::MetalMatrixHandle;
use std::ops::{ Add, AddAssign, Sub, SubAssign, Mul, MulAssign };
use serde::{ Deserializer, Deserialize, Serializer, Serialize };

pub struct VerifyingMatrixHandle {
    pub simple: SimpleMatrixHandle,
    pub metal: MetalMatrixHandle,
}

impl VerifyingMatrixHandle {
    pub fn verify(&self) {
        const epsilon: f32 = 0.0001;

        let simple_matrix = SimpleMatrixHandle::to_matrix(&self.simple);
        let metal_matrix = MetalMatrixHandle::to_matrix(&self.metal);

        if simple_matrix.rows() != metal_matrix.rows() {
            panic!("Different number of rows. Expected {} found {}", simple_matrix.rows(), metal_matrix.rows());
        }
        if simple_matrix.columns() != metal_matrix.columns() {
            panic!("Different number of columns. Expected {} found {}", simple_matrix.columns(), metal_matrix.columns());
        }

        let mut is_approx_equal = true;
        for i in 0..simple_matrix.rows() {
            for j in 0..simple_matrix.columns() {
                if (simple_matrix[(i, j)] - metal_matrix[(i, j)]).abs() > epsilon {
                    is_approx_equal = false;
                    break;
                }
            }
        }

        if !is_approx_equal {
            println!("Difference:\n{}", &simple_matrix - &metal_matrix);
            panic!("Failed to verify matrix handles!\nExpected:\n{}\n\nFound:\n{}", simple_matrix, metal_matrix);
        }
    }
}

impl MatrixHandle for VerifyingMatrixHandle {
    fn of_size(rows: usize, columns: usize) -> VerifyingMatrixHandle {
        return VerifyingMatrixHandle {
            simple: SimpleMatrixHandle::of_size(rows, columns),
            metal: MetalMatrixHandle::of_size(rows, columns)
        }
    }

    fn from_matrix(matrix: &Matrix<f32>) -> VerifyingMatrixHandle {
        return VerifyingMatrixHandle {
            simple: SimpleMatrixHandle::from_matrix(&matrix),
            metal: MetalMatrixHandle::from_matrix(&matrix)
        }
    }

    fn copy_from_matrix(dst: &mut Self, matrix: Matrix<f32>) {
        panic!("'copy_from_matrix' not implemented!");
    }

    fn to_matrix(&self) -> Matrix<f32> {
        self.verify();
        return self.simple.to_matrix();
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
            simple: self.simple.add_constant_row(value),
            metal: self.metal.add_constant_row(value)
        };
        res.verify();
        return res;
    }

    fn remove_first_row(&self) -> Self {
        let res = VerifyingMatrixHandle {
            simple: self.simple.remove_first_row(),
            metal: self.metal.remove_first_row()
        };
        res.verify();
        return res;
    }

    fn transpose(&self) -> Self {
        self.verify();
        let res = VerifyingMatrixHandle {
            simple: self.simple.transpose(),
            metal: self.metal.transpose()
        };
        res.verify();
        return res;
    }

    fn entrywise_product(&self, rhs: &VerifyingMatrixHandle) -> Self {
        let metal_lhs = MetalMatrixHandle::to_matrix(&self.metal);
        let metal_rhs = MetalMatrixHandle::to_matrix(&rhs.metal);

        let res = VerifyingMatrixHandle {
            simple: self.simple.entrywise_product(&rhs.simple),
            metal: self.metal.entrywise_product(&rhs.metal)
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
        return self.simple.rows();
    }

    fn columns(&self) -> usize {
        self.verify();
        return self.simple.columns();
    }
}

impl Clone for VerifyingMatrixHandle {
    fn clone(&self) -> VerifyingMatrixHandle {
        let res = VerifyingMatrixHandle {
            simple: self.simple.clone(),
            metal: self.metal.clone()
        };
        res.verify();
        return res;
    }
}

impl Add for VerifyingMatrixHandle {
    type Output = VerifyingMatrixHandle;
    fn add(self, rhs: VerifyingMatrixHandle) -> VerifyingMatrixHandle {
        return &self + &rhs;
    }
}
impl<'a> Add<&'a VerifyingMatrixHandle> for &'a VerifyingMatrixHandle {
    type Output = VerifyingMatrixHandle;

    fn add(self, rhs: &'a VerifyingMatrixHandle) -> VerifyingMatrixHandle {
        let res = VerifyingMatrixHandle {
            simple: &self.simple + &rhs.simple,
            metal: &self.metal + &rhs.metal
        };
        res.verify();
        return res;
    }
}

impl AddAssign for VerifyingMatrixHandle {
    fn add_assign(&mut self, rhs: VerifyingMatrixHandle) {
        self.simple += rhs.simple;
        self.metal += rhs.metal;
        self.verify();
    }
}

impl Sub for VerifyingMatrixHandle {
    type Output = VerifyingMatrixHandle;

    fn sub(self, rhs: VerifyingMatrixHandle) -> VerifyingMatrixHandle {
        return &self - &rhs;
    }
}
impl<'a> Sub<&'a VerifyingMatrixHandle> for &'a VerifyingMatrixHandle {
    type Output = VerifyingMatrixHandle;

    fn sub(self, rhs: &'a VerifyingMatrixHandle) -> VerifyingMatrixHandle {
        let res = VerifyingMatrixHandle {
            simple: &self.simple - &rhs.simple,
            metal: &self.metal - &rhs.metal
        };
        res.verify();
        return res;
    }
}

impl SubAssign<VerifyingMatrixHandle> for VerifyingMatrixHandle {
    fn sub_assign(&mut self, rhs: VerifyingMatrixHandle) {
        panic!("not implemented");
    }
}
impl SubAssign<&VerifyingMatrixHandle> for VerifyingMatrixHandle {
    fn sub_assign(&mut self, rhs: &VerifyingMatrixHandle) {
        panic!("not implemented");
    }
}

impl Mul for VerifyingMatrixHandle {
    type Output = VerifyingMatrixHandle;
    fn mul(self, rhs: VerifyingMatrixHandle) -> VerifyingMatrixHandle {
        return &self * &rhs;
    }
}
impl<'a> Mul for &'a VerifyingMatrixHandle {
    type Output = VerifyingMatrixHandle;
    fn mul(self, rhs: &VerifyingMatrixHandle) -> VerifyingMatrixHandle {
        let res = VerifyingMatrixHandle {
            simple: &self.simple * &rhs.simple,
            metal: &self.metal * &rhs.metal
        };
        res.verify();
        return res;
    }
}

impl<'a> Mul<f32> for &'a VerifyingMatrixHandle {
    type Output = VerifyingMatrixHandle;
    fn mul(self, scalar: f32) -> VerifyingMatrixHandle {
        let res = VerifyingMatrixHandle {
            simple: &self.simple * scalar,
            metal: &self.metal * scalar
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
