use matrix::matrix::Matrix;
use matrix::matrixhandle::MatrixHandle;

pub trait ActivationFunction<T>: Send + Sync {
    fn evaluate(&self, &T) -> T;
    fn derivative(&self, &T) -> T;
    fn inline_evaluate(&self, &mut T);
    fn inline_derivative(&self, &mut T);
}

pub struct Sigmoid;
pub struct Relu;
pub struct TwoPlayerScore;
