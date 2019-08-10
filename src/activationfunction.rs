use simplematrixhandle::SimpleMatrixHandle;
use matrix::Matrix;
use matrixhandle::MatrixHandle;

#[cfg(matrixlib)]
#[link(name = "matrix", kind = "static")]
extern {
    fn matrix_apply_sigmoid(handle_a: *const LibMatrixHandle,
                            handle_result: *mut LibMatrixHandle) -> libc::c_int;
    fn matrix_apply_sigmoid_derivative(handle_a: *const LibMatrixHandle,
                                       handle_result: *mut LibMatrixHandle) -> libc::c_int;
    
    fn matrix_apply_relu(handle_a: *const LibMatrixHandle,
                         handle_result: *mut LibMatrixHandle) -> libc::c_int;
    fn matrix_apply_relu_derivative(handle_a: *const LibMatrixHandle,
                                    handle_result: *mut LibMatrixHandle) -> libc::c_int;
    
    fn matrix_apply_twoplayerscore(handle_a: *const LibMatrixHandle,
                                   handle_result: *mut LibMatrixHandle) -> libc::c_int;
    fn matrix_apply_twoplayerscore_derivative(handle_a: *const LibMatrixHandle,
                                              handle_result: *mut LibMatrixHandle) -> libc::c_int;
}

pub trait ActivationFunction<T>: Send + Sync {
    fn evaluate(&self, &T) -> T;
    fn derivative(&self, &T) -> T;
    fn inline_evaluate(&self, &mut T);
    fn inline_derivative(&self, &mut T);
}

pub struct Sigmoid;
impl ActivationFunction<f32> for Sigmoid {
    fn evaluate(&self, x: &f32) -> f32 {
        let exp = x.exp();
        if exp.is_infinite() {
            return 1f32;
        }

        let res = exp / (exp + 1_f32);
        if cfg!(debug_assertions) && res.is_nan() {
            panic!("Evaluated to NaN!");
        }
        return res;
    }

    fn derivative(&self, x: &f32) -> f32 {
        let exp = x.exp();
        if exp.is_infinite() {
            return 0f32;
        }

        let res = exp / (exp + 1_f32).powi(2);
        if cfg!(debug_assertions) && res.is_nan() {
            panic!("Evaluated to NaN!");
        }
        return res;
    }

    fn inline_evaluate(&self, x: &mut f32) {
        *x = self.evaluate(&x);
    }

    fn inline_derivative(&self, x: &mut f32) {
        *x = self.derivative(&x);
    }
}

impl ActivationFunction<Matrix<f32>> for Sigmoid {
    fn evaluate(&self, input: &Matrix<f32>) -> Matrix<f32> {
        return Matrix::new(input.rows(), input.columns(), &|row, column| {
            return self.evaluate(&input[(row, column)]);
        });
    }

    fn derivative(&self, input: &Matrix<f32>) -> Matrix<f32> {
        return Matrix::new(input.rows(), input.columns(), &|row, column| {
            return self.derivative(&input[(row, column)]);
        });
    }
    
    fn inline_evaluate(&self, input: &mut Matrix<f32>) {
        for row in 0 .. input.rows() {
            for column in 0 .. input.columns() {
                input[(row, column)] = self.evaluate(&input[(row, column)]);
            }
        }
    }

    fn inline_derivative(&self, input: &mut Matrix<f32>) {
        for row in 0 .. input.rows() {
            for column in 0 .. input.columns() {
                input[(row, column)] = self.derivative(&input[(row, column)]);
            }
        }
    }
}

impl ActivationFunction<SimpleMatrixHandle> for Sigmoid {
    fn evaluate(&self, input: &SimpleMatrixHandle) -> SimpleMatrixHandle {
        return SimpleMatrixHandle::from_matrix(self.evaluate(&SimpleMatrixHandle::to_matrix(input)));
    }

    fn derivative(&self, input: &SimpleMatrixHandle) -> SimpleMatrixHandle {
        return SimpleMatrixHandle::from_matrix(self.derivative(&SimpleMatrixHandle::to_matrix(input)));
    }

    fn inline_evaluate(&self, input: &mut SimpleMatrixHandle) {
        panic!("Not implemented!");
    }

    fn inline_derivative(&self, input: &mut SimpleMatrixHandle) {
        panic!("Not implemented!");
    }
}

#[cfg(matrixlib)]
impl ActivationFunction<MatrixHandle> for Sigmoid {
    fn evaluate(&self, input: &MatrixHandle) -> MatrixHandle {
        let mut result_handle = MatrixHandle::empty();
        let result = unsafe {
            matrix_apply_sigmoid(input as *const MatrixHandle,
                                 &mut result_handle as *mut MatrixHandle)
        };
        if result != 0 {
            panic!("Failed to calculate sigmoid!");
        }
        return result_handle;
    }


    fn derivative(&self, input: &MatrixHandle) -> MatrixHandle {
        let mut result_handle = MatrixHandle::empty();
        let result = unsafe {
            matrix_apply_sigmoid_derivative(input as *const MatrixHandle,
                                            &mut result_handle as *mut MatrixHandle)
        };
        if result != 0 {
            panic!("Failed to calculate sigmoid derivative!");
        }
        return result_handle;
    }

    fn inline_evaluate(&self, input: &mut MatrixHandle) {
        let result = unsafe {
            matrix_apply_sigmoid(input as *const MatrixHandle,
                                 input as *mut MatrixHandle)
        };
        if result != 0 {
            panic!("Failed to calculate inline sigmoid")
        }
    }

    fn inline_derivative(&self, input: &mut MatrixHandle) {
        let result = unsafe {
            matrix_apply_sigmoid_derivative(input as *const MatrixHandle,
                                            input as *mut MatrixHandle)
        };
        if result != 0 {
            panic!("Failed to calculate inline sigmoid")
        }
    }
}

pub struct Relu;
impl ActivationFunction<f32> for Relu {
    fn evaluate(&self, x: &f32) -> f32 {
        return if *x < 0_f32 { 0_f32 } else { *x };
    }

    fn derivative(&self, x: &f32) -> f32 {
        return if *x < 0_f32 { 0_f32 } else { 1_f32 };
    }
    
    fn inline_evaluate(&self, x: &mut f32) {
        *x = self.evaluate(&x);
    }

    fn inline_derivative(&self, x: &mut f32) {
        *x = self.derivative(&x);
    }
}

impl ActivationFunction<Matrix<f32>> for Relu {
    fn evaluate(&self, input: &Matrix<f32>) -> Matrix<f32> {
        return Matrix::new(input.rows(), input.columns(), &|row, column| {
            return self.evaluate(&input[(row, column)]);
        });
    }

    fn derivative(&self, input: &Matrix<f32>) -> Matrix<f32> {
        return Matrix::new(input.rows(), input.columns(), &|row, column| {
            return self.derivative(&input[(row, column)]);
        });
    }

    fn inline_evaluate(&self, input: &mut Matrix<f32>) {
        for row in 0 .. input.rows() {
            for column in 0 .. input.columns() {
                input[(row, column)] = self.evaluate(&input[(row, column)]);
            }
        }
    }

    fn inline_derivative(&self, input: &mut Matrix<f32>) {
        for row in 0 .. input.rows() {
            for column in 0 .. input.columns() {
                input[(row, column)] = self.derivative(&input[(row, column)]);
            }
        }
    }
}

impl ActivationFunction<SimpleMatrixHandle> for Relu {
    fn evaluate(&self, input: &SimpleMatrixHandle) -> SimpleMatrixHandle {
        return SimpleMatrixHandle::from_matrix(self.evaluate(&SimpleMatrixHandle::to_matrix(input)));
    }

    fn derivative(&self, input: &SimpleMatrixHandle) -> SimpleMatrixHandle {
        return SimpleMatrixHandle::from_matrix(self.derivative(&SimpleMatrixHandle::to_matrix(input)));
    }

    fn inline_evaluate(&self, input: &mut SimpleMatrixHandle) {
        panic!("Not implemented!");
    }

    fn inline_derivative(&self, input: &mut SimpleMatrixHandle) {
        panic!("Not implemented!");
    }
}

#[cfg(matrixlib)]
impl ActivationFunction<MatrixHandle> for Relu {
    fn evaluate(&self, input: &MatrixHandle) -> MatrixHandle {
        let mut result_handle = MatrixHandle::empty();
        let result = unsafe {
            matrix_apply_relu(input as *const MatrixHandle,
                              &mut result_handle as *mut MatrixHandle)
        };
        if result != 0 {
            panic!("Failed to transpose matrices!");
        }
        return result_handle;
    }


    fn derivative(&self, input: &MatrixHandle) -> MatrixHandle {
        let mut result_handle = MatrixHandle::empty();
        let result = unsafe {
            matrix_apply_relu_derivative(input as *const MatrixHandle,
                                         &mut result_handle as *mut MatrixHandle)
        };
        if result != 0 {
            panic!("Failed to transpose matrices!");
        }
        return result_handle;
    }
    
    fn inline_evaluate(&self, input: &mut MatrixHandle) {
        let result = unsafe {
            matrix_apply_relu(input as *const MatrixHandle,
                              input as *mut MatrixHandle)
        };
        if result != 0 {
            panic!("Failed to calculate inline sigmoid")
        }
    }

    fn inline_derivative(&self, input: &mut MatrixHandle) {
        let result = unsafe {
            matrix_apply_relu_derivative(input as *const MatrixHandle,
                                         input as *mut MatrixHandle)
        };
        if result != 0 {
            panic!("Failed to calculate inline sigmoid")
        }
    }
}

pub struct TwoPlayerScore;
impl ActivationFunction<f32> for TwoPlayerScore {
    fn evaluate(&self, x: &f32) -> f32 {
        let exp = x.exp();
        if exp.is_infinite() {
            return 1f32;
        }

        let res = (exp - 1_f32) / (exp + 1_f32);
        if cfg!(debug_assertions) && res.is_nan() {
            panic!("Evaluated to NaN!");
        }
        return res;
    }

    fn derivative(&self, x: &f32) -> f32 {
        let exp = x.exp();
        if exp.is_infinite() {
            return 0f32;
        }

        let res = 2_f32 * exp / (exp + 1_f32).powi(2);
        if cfg!(debug_assertions) && res.is_nan() {
            panic!("Evaluated to NaN!");
        }
        return res;
    }

    fn inline_evaluate(&self, x: &mut f32) {
        *x = self.evaluate(&x);
    }

    fn inline_derivative(&self, x: &mut f32) {
        *x = self.derivative(&x);
    }
}

impl ActivationFunction<Matrix<f32>> for TwoPlayerScore {
    fn evaluate(&self, input: &Matrix<f32>) -> Matrix<f32> {
        return Matrix::new(input.rows(), input.columns(), &|row, column| {
            return self.evaluate(&input[(row, column)]);
        });
    }

    fn derivative(&self, input: &Matrix<f32>) -> Matrix<f32> {
        return Matrix::new(input.rows(), input.columns(), &|row, column| {
            return self.derivative(&input[(row, column)]);
        });
    }

    fn inline_evaluate(&self, input: &mut Matrix<f32>) {
        for row in 0 .. input.rows() {
            for column in 0 .. input.columns() {
                input[(row, column)] = self.evaluate(&input[(row, column)]);
            }
        }
    }

    fn inline_derivative(&self, input: &mut Matrix<f32>) {
        for row in 0 .. input.rows() {
            for column in 0 .. input.columns() {
                input[(row, column)] = self.derivative(&input[(row, column)]);
            }
        }
    }
}

impl ActivationFunction<SimpleMatrixHandle> for TwoPlayerScore {
    fn evaluate(&self, input: &SimpleMatrixHandle) -> SimpleMatrixHandle {
        return SimpleMatrixHandle::from_matrix(self.evaluate(&SimpleMatrixHandle::to_matrix(input)));
    }

    fn derivative(&self, input: &SimpleMatrixHandle) -> SimpleMatrixHandle {
        return SimpleMatrixHandle::from_matrix(self.derivative(&SimpleMatrixHandle::to_matrix(input)));
    }

    fn inline_evaluate(&self, input: &mut SimpleMatrixHandle) {
        panic!("Not implemented!");
    }

    fn inline_derivative(&self, input: &mut SimpleMatrixHandle) {
        panic!("Not implemented!");
    }
}

#[cfg(matrixlib)]
impl ActivationFunction<MatrixHandle> for TwoPlayerScore {
    fn evaluate(&self, input: &MatrixHandle) -> MatrixHandle {
        let mut result_handle = MatrixHandle::empty();
        let result = unsafe {
            matrix_apply_twoplayerscore(input as *const MatrixHandle,
                                        &mut result_handle as *mut MatrixHandle)
        };
        if result != 0 {
            panic!("Failed to transpose matrices!");
        }
        return result_handle;
    }


    fn derivative(&self, input: &MatrixHandle) -> MatrixHandle {
        let mut result_handle = MatrixHandle::empty();
        let result = unsafe {
            matrix_apply_twoplayerscore_derivative(input as *const MatrixHandle,
                                                   &mut result_handle as *mut MatrixHandle)
        };
        if result != 0 {
            panic!("Failed to transpose matrices!");
        }
        return result_handle;
    }
    
    fn inline_evaluate(&self, input: &mut MatrixHandle) {
        let result = unsafe {
            matrix_apply_twoplayerscore(input as *const MatrixHandle,
                                        input as *mut MatrixHandle)
        };
        if result != 0 {
            panic!("Failed to calculate inline sigmoid")
        }
    }

    fn inline_derivative(&self, input: &mut MatrixHandle) {
        let result = unsafe {
            matrix_apply_twoplayerscore_derivative(input as *const MatrixHandle,
                                                   input as *mut MatrixHandle)
        };
        if result != 0 {
            panic!("Failed to calculate inline sigmoid")
        }
    }
}