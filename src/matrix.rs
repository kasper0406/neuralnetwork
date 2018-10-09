use std::fmt;
use std::ops::{Add, AddAssign, Sub, Mul, Index, IndexMut};
use rand::{thread_rng, Rng};
use rand::distributions::Uniform;
use num::Zero;
use std::collections::HashSet;
use std::cmp;

use rayon::prelude::*;

#[derive(Clone, Serialize, Deserialize)]
pub struct Matrix<T> {
    rows: usize,
    columns: usize,
    values: Vec<T>
}

impl<T: Copy + Mul<Output=T> + Zero + Send + Sync + Send + Sync + AddAssign + Add<Output=T>> Matrix<T> {
    pub fn new<F: Fn(usize, usize) -> T + Sync>(rows: usize, columns: usize, populator: &F) -> Matrix<T> {
        assert!(rows > 0, "A matrix must have >0 rows");
        assert!(columns > 0, "A matrix must have >0 columns");

        let mut values = Vec::with_capacity(columns * rows);
        for row in 0..rows {
            for column in 0..columns {
                values.push(populator(row, column));
            }
        }

        Matrix { rows: rows, columns: columns, values: values }
    }

    pub fn slow_mul(self, rhs: &Matrix<T>) -> Matrix<T> {
        assert!(self.columns == rhs.rows, "Columns in lhs must match rows in rhs");

        return Matrix::new(self.rows, rhs.columns, &|row, column| {
            let mut res: T = T::zero();
            for i in 0..self.columns {
                res += self[(row, i)] * rhs[(i, column)];
            }

            return res;
        });
    }

    pub fn transpose(&self) -> Matrix<T> {
        Matrix::new(self.columns, self.rows, &|row, column| {
            return self[(column, row)];
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn size(&self) -> (usize, usize) {
        return (self.rows(), self.columns());
    }

    pub fn entrywise_product(&self, other: &Matrix<T>) -> Matrix<T> {
        assert!(self.size() == other.size(), "Matrices must have the same size!");
        Matrix::new(self.rows(), self.columns(), &|row, column| {
            return self[(row, column)] * other[(row, column)];
        })
    }

    pub fn add_constant_row(&self, value: T) -> Matrix<T> {
        return Matrix::new(self.rows() + 1, self.columns(), &|row, column| {
            if row == 0 {
                return value;
            } else {
                return self[(row - 1, column)];
            }
        });
    }
    
    pub fn remove_first_row(&self) -> Matrix<T> {
        return Matrix::new(self.rows() - 1, self.columns(), &|row, column| {
            return self[(row + 1, column)];
        });
    }

    pub fn dropout_elements(&self, rate: f64) -> Matrix<T> {
        assert!(0_f64 <= rate && rate < 1_f64, "Dropout rate must be in interval [0; 1[");

        let distr = Uniform::new(0_f64, 1_f64);
        return Matrix::new(self.rows(), self.columns(), &|row, column| {
            if thread_rng().sample(distr) < rate {
                return T::zero();
            } else {
                return self[(row, column)];
            }
        });
    }

    pub fn dropout_rows(&self, rate: f64) -> Matrix<T> {
        assert!(0_f64 <= rate && rate < 1_f64, "Dropout rate must be in interval [0; 1[");

        let distr = Uniform::new(0_f64, 1_f64);
        let mut rows_to_drop = HashSet::new();
        for row in 0..self.rows {
            if thread_rng().sample(distr) < rate {
                rows_to_drop.insert(row);
            }
        }
        
        return Matrix::new(self.rows(), self.columns(), &|row, column| {
            if rows_to_drop.contains(&row) {
                return T::zero();
            } else {
                return self[(row, column)];
            }
        });
    }
}

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index<'a>(&'a self, index: (usize, usize)) -> &'a T {
        return &self.values[self.columns * index.0 + index.1];
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut<'a>(&'a mut self, index: (usize, usize)) -> &'a mut T {
        return &mut self.values[self.columns * index.0 + index.1];
    }
}

impl<'a, T: Add<Output=T> + Mul<Output=T> + AddAssign + Zero + Send + Sync + Copy> Add for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, other: &Matrix<T>) -> Matrix<T> {
        assert!(self.rows == other.rows, "Row count must be the same!");
        assert!(self.columns == other.columns, "Column count must be the same!");

        return Matrix::new(self.rows, self.columns, &|row: usize, column: usize| {
            return self[(row, column)] + other[(row, column)];
        });
    }
}

impl<T: AddAssign + Copy> AddAssign for Matrix<T> {
    fn add_assign(&mut self, other: Matrix<T>) {
        assert!(self.rows == other.rows, "Row count must be the same!");
        assert!(self.columns == other.columns, "Column count must be the same!");

        for row in 0..self.rows {
            for column in 0..self.columns {
                self[(row, column)] += other[(row, column)];
            }
        }
    }
}

impl<'a, T: Sub<Output=T> + Mul<Output=T> + AddAssign + Zero + Send + Sync + Copy> Sub for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn sub(self, other: &Matrix<T>) -> Matrix<T> {
        assert!(self.rows == other.rows, "Row count must be the same!");
        assert!(self.columns == other.columns, "Column count must be the same!");

        return Matrix::new(self.rows, self.columns, &|row: usize, column: usize| {
            return self[(row, column)] - other[(row, column)];
        });
    }
}

impl<'a, T: Mul<Output=T> + AddAssign + Add<Output=T> + Copy + Zero + Send + Sync> Mul for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, rhs: &Matrix<T>) -> Matrix<T> {
        assert!(self.columns == rhs.rows, "Columns in lhs must match rows in rhs");

        let mut values = vec![T::zero(); self.rows * rhs.columns];

        let column_multiplier = cmp::max(1, 30 / rhs.columns);
        let chunk_size = cmp::min(values.len(), column_multiplier * rhs.columns);
        // println!("Multiplier = {}, chunk_size = {}", column_multiplier, chunk_size);
        values.as_mut_slice()
                .par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(chunk_num, chunk)| {
                    let chunk_row_start = chunk_num * chunk_size / rhs.columns;

                    for chunk_index in 0..chunk.len() {
                        let row_num = chunk_row_start + chunk_index / rhs.columns;
                        let col_num = chunk_index % rhs.columns;

                        let mut res: T = T::zero();
                        for k in 0..self.columns {
                            res += self[(row_num, k)] * rhs[(k, col_num)];
                        }
                        chunk[chunk_index] = res;
                    }
                });
        
        Matrix { rows: self.rows, columns: rhs.columns, values: values }
    }
}

impl<'a> Mul<f64> for &'a Matrix<f64> {
    type Output = Matrix<f64>;

    fn mul(self, rhs: f64) -> Matrix<f64> {
        return Matrix::new(self.rows, self.columns, &|row, column| {
            return rhs * self[(row, column)];
        });
    }
}

impl<'a> Mul<&'a Matrix<f64>> for f64 {
    type Output = Matrix<f64>;

    fn mul(self, rhs: &Matrix<f64>) -> Matrix<f64> {
        return rhs * self;
    }
}

impl<'a, T: Mul<Output=T> + AddAssign + Add<Output=T> + Copy + Zero + Send + Sync> Mul for Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, rhs: Matrix<T>) -> Matrix<T> {
        return &self * &rhs;
    }
}

impl<T: fmt::Display> fmt::Display for Matrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut result = write!(f, "");
        for row in 0..self.rows {
            for col in 0..self.columns {
                result = result.and(write!(f, "{} ", self[(row, col)]));
            }
            result = result.and(write!(f, "\n"));
        }
        return result;
    }
}
