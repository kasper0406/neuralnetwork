use serde::{ Deserializer, Deserialize, Serializer, Serialize };
use matrix::Matrix;
use std::ops::{Add, AddAssign, Sub, SubAssign, Mul};

pub trait MatrixHandle: Add + AddAssign + Sub + SubAssign + Mul + Mul<f32> + Clone
{
    fn of_size(rows: usize, columns: usize) -> Self;
    fn from_matrix(matrix: &Matrix<f32>) -> Self;
    fn to_matrix(&self) -> Matrix<f32>;
    fn copy_from_matrix(dst: &mut Self, matrix: Matrix<f32>);

    fn copy(destination: &mut Self, source: &Self);

    fn dropout_elements(&self, rate: f32) -> Self;
    fn dropout_rows(&self, rate: f32) -> Self;
    fn add_constant_row(&self, value: f32) -> Self;
    fn remove_first_row(&self) -> Self;
    fn transpose(&self) -> Self;
    fn entrywise_product(&self, rhs: &Self) -> Self;

    fn inplace_entrywise_product(&mut self, rhs: &Self);
    fn inplace_add_constant_row(&mut self, value: f32);
    fn inplace_remove_first_row(&mut self);
    fn inplace_transpose(&mut self);
    fn inplace_scalar_multiply(&mut self, scalar: f32);

    fn multiply(lhs: &Self, rhs: &Self, dst: &mut Self);

    fn rows(&self) -> usize;
    fn columns(&self) -> usize;
}
