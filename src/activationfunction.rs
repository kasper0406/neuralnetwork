use matrix::Matrix;

pub trait ActivationFunction<T> {
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
