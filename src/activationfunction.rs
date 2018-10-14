use matrix::Matrix;
use matrixhandle::MatrixHandle;

#[link(name = "matrix", kind = "static")]
extern {
    fn matrix_apply_sigmoid(handle_a: *const MatrixHandle,
                            handle_result: *mut MatrixHandle) -> libc::c_int;
    fn matrix_apply_sigmoid_derivative(handle_a: *const MatrixHandle,
                                       handle_result: *mut MatrixHandle) -> libc::c_int;
    
    fn matrix_apply_relu(handle_a: *const MatrixHandle,
                         handle_result: *mut MatrixHandle) -> libc::c_int;
    fn matrix_apply_relu_derivative(handle_a: *const MatrixHandle,
                                    handle_result: *mut MatrixHandle) -> libc::c_int;
    
    fn matrix_apply_twoplayerscore(handle_a: *const MatrixHandle,
                                   handle_result: *mut MatrixHandle) -> libc::c_int;
    fn matrix_apply_twoplayerscore_derivative(handle_a: *const MatrixHandle,
                                              handle_result: *mut MatrixHandle) -> libc::c_int;
}

pub trait ActivationFunction<T>: Send + Sync {
    fn evaluate(&self, &T) -> T;
    fn derivative(&self, &T) -> T;
}

pub struct Sigmoid;
impl ActivationFunction<f64> for Sigmoid {
    fn evaluate(&self, x: &f64) -> f64 {
        return x.exp() / (x.exp() + 1_f64);
    }

    fn derivative(&self, x: &f64) -> f64 {
        return x.exp() / (x.exp() + 1_f64).powi(2);
    }
}

impl ActivationFunction<Matrix<f64>> for Sigmoid {
    fn evaluate(&self, input: &Matrix<f64>) -> Matrix<f64> {
        return Matrix::new(input.rows(), input.columns(), &|row, column| {
            return self.evaluate(&input[(row, column)]);
        });
    }

    fn derivative(&self, input: &Matrix<f64>) -> Matrix<f64> {
        return Matrix::new(input.rows(), input.columns(), &|row, column| {
            return self.derivative(&input[(row, column)]);
        });
    }
}

impl ActivationFunction<MatrixHandle> for Sigmoid {
    fn evaluate(&self, input: &MatrixHandle) -> MatrixHandle {
        let mut result_handle = MatrixHandle::empty();
        let result = unsafe {
            matrix_apply_sigmoid(input as *const MatrixHandle,
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
            matrix_apply_sigmoid_derivative(input as *const MatrixHandle,
                                            &mut result_handle as *mut MatrixHandle)
        };
        if result != 0 {
            panic!("Failed to transpose matrices!");
        }
        return result_handle;
    }
}

pub struct Relu;
impl ActivationFunction<f64> for Relu {
    fn evaluate(&self, x: &f64) -> f64 {
        return if *x < 0_f64 { 0_f64 } else { *x };
    }

    fn derivative(&self, x: &f64) -> f64 {
        return if *x < 0_f64 { 0_f64 } else { 1_f64 };
    }
}

impl ActivationFunction<Matrix<f64>> for Relu {
    fn evaluate(&self, input: &Matrix<f64>) -> Matrix<f64> {
        return Matrix::new(input.rows(), input.columns(), &|row, column| {
            return self.evaluate(&input[(row, column)]);
        });
    }

    fn derivative(&self, input: &Matrix<f64>) -> Matrix<f64> {
        return Matrix::new(input.rows(), input.columns(), &|row, column| {
            return self.derivative(&input[(row, column)]);
        });
    }
}

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
}

pub struct TwoPlayerScore;
impl ActivationFunction<f64> for TwoPlayerScore {
    fn evaluate(&self, x: &f64) -> f64 {
        return (x.exp() - 1_f64) / (x.exp() + 1_f64);
    }

    fn derivative(&self, x: &f64) -> f64 {
        return 2_f64 * x.exp() / (x.exp() + 1_f64).powi(2);
    }
}

impl ActivationFunction<Matrix<f64>> for TwoPlayerScore {
    fn evaluate(&self, input: &Matrix<f64>) -> Matrix<f64> {
        return Matrix::new(input.rows(), input.columns(), &|row, column| {
            return self.evaluate(&input[(row, column)]);
        });
    }

    fn derivative(&self, input: &Matrix<f64>) -> Matrix<f64> {
        return Matrix::new(input.rows(), input.columns(), &|row, column| {
            return self.derivative(&input[(row, column)]);
        });
    }
}

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
}