use serde::{ Deserializer, Deserialize, Serializer, Serialize };
use matrix::Matrix;
use std::ops::{Add, AddAssign, Sub, SubAssign, Mul};
use std::ptr;

#[cfg(matrixlib)]
#[repr(C)]
pub struct MatrixLibHandle {
    rows: usize,
    columns: usize,
    elements: *const f32,
    allocated_bytes: usize,
    base_ptr: *const f32
}

#[cfg(matrixlib)]
unsafe impl Send for MatrixLibHandle { }

#[cfg(matrixlib)]
#[link(name = "matrix", kind = "static")]
extern {

    fn matrix_synchronize(only_current_thread: u8);

    fn matrix_alloc(rows: libc::size_t,
                    columns: libc::size_t,
                    elements: *const libc::c_float,
                    handle: *mut MatrixHandle) -> libc::c_int;

    fn matrix_alloc_or_reuse(handle: *mut MatrixHandle,
                                rows: libc::size_t, columns: libc::size_t) -> libc::c_int;

    fn matrix_free(handle: *mut MatrixHandle);

    fn matrix_device_to_host(handle: *const MatrixHandle,
                             host_elements: *mut libc::c_float) -> libc::c_int;

    fn matrix_add(handle_a: *const MatrixHandle,
                  handle_b: *const MatrixHandle,
                  handle_result: *mut MatrixHandle) -> libc::c_int;

    fn matrix_add_assign(handle_a: *mut MatrixHandle,
                         handle_b: *const MatrixHandle) -> libc::c_int;

    fn matrix_sub(handle_a: *const MatrixHandle,
                  handle_b: *const MatrixHandle,
                  handle_result: *mut MatrixHandle) -> libc::c_int;

    fn matrix_sub_assign(handle_a: *mut MatrixHandle,
                         handle_b: *const MatrixHandle) -> libc::c_int;

    fn matrix_entrywise_multiply(handle_a: *const MatrixHandle,
                                 handle_b: *const MatrixHandle,
                                 handle_result: *mut MatrixHandle) -> libc::c_int;

    fn matrix_inplace_entrywise_multiply(handle_a: *mut MatrixHandle,
                                         handle_b: *const MatrixHandle) -> libc::c_int;

    fn matrix_scalar_multiply(handle_a: *const MatrixHandle,
                              scalar: libc::c_float,
                              handle_result: *mut MatrixHandle) -> libc::c_int;

    fn matrix_multiply(handle_a: *const MatrixHandle,
                       handle_b: *const MatrixHandle,
                       handle_result: *mut MatrixHandle) -> libc::c_int;
    
    fn matrix_transpose(handle_a: *const MatrixHandle,
                        handle_result: *mut MatrixHandle) -> libc::c_int;

    fn matrix_inplace_transpose(handle: *mut MatrixHandle) -> libc::c_int;

    fn matrix_add_constant_row(padding: f32,
                               handle_a: *const MatrixHandle,
                               handle_result: *mut MatrixHandle) -> libc::c_int;

    fn matrix_inplace_add_constant_row(padding: f32, handle: *mut MatrixHandle) -> libc::c_int;

    fn matrix_inplace_remove_first_row(handle: *mut MatrixHandle) -> libc::c_int;

    fn matrix_dropout_elements(rate: f32,
                               handle_a: *const MatrixHandle,
                               handle_result: *mut MatrixHandle) -> libc::c_int;

    fn matrix_dropout_rows(rate: f32,
                           handle_a: *const MatrixHandle,
                           handle_result: *mut MatrixHandle) -> libc::c_int;
    
    fn matrix_copy(handle_a: *const MatrixHandle,
                   handle_result: *mut MatrixHandle) -> libc::c_int;
}

#[cfg(matrixlib)]
impl MatrixLibHandle {
    pub fn empty() -> MatrixHandle {
        MatrixHandle {
            rows: 0,
            columns: 0,
            elements: ptr::null(),
            allocated_bytes: 0,
            base_ptr: ptr::null()
        }
    }

    pub fn synchronize(only_current_thread: bool) {
        unsafe {
            matrix_synchronize(match only_current_thread {
                false => 0,
                true => 1
            });
        }
    }
}

#[cfg(matrixlib)]
impl MatrixHandle for MatrixLibHandle {
    pub fn copy(destination: &mut MatrixHandle, source: &MatrixHandle) {
        let copy_result = unsafe { matrix_copy(source, destination) };
        if copy_result != 0 {
            panic!("Failed to copy matrices!");
        }
    }

    pub fn from_matrix(matrix: Matrix<f32>) -> MatrixHandle {
        let mut handle: MatrixHandle = MatrixHandle::empty();

        let alloc_result = unsafe {
            matrix_alloc(matrix.rows(),
                         matrix.columns(),
                         matrix.raw_values().as_ptr(),
                         &mut handle as *mut MatrixHandle)
        };
        if alloc_result != 0 {
            panic!("Failed to create MatrixHandle: {}", alloc_result);
        }

        return handle;
    }

    pub fn copy_from_matrix(dst: &mut MatrixHandle, matrix: Matrix<f32>) {
        let result = unsafe {
            matrix_alloc(matrix.rows(),
                         matrix.columns(),
                         matrix.raw_values().as_ptr(),
                         dst as *mut MatrixHandle)
        };
        if result != 0 {
            panic!("Failed to copy from matrix: {}", result);
        }
    }

    pub fn to_matrix(handle: &MatrixHandle) -> Matrix<f32> {
        MatrixHandle::synchronize(true);

        let mut elements = vec![0_f32; handle.rows * handle.columns];
        let device_to_host_result = unsafe {
            matrix_device_to_host(handle as *const MatrixHandle,
                                  elements.as_mut_ptr())
        };
        if device_to_host_result != 0 {
            panic!("Failed to transfer device memory to host!");
        }

        return Matrix::new(handle.rows, handle.columns, &|row, column| {
            elements[row * handle.columns + column]
        });
    }

    pub fn of_size(rows: usize, columns: usize) -> MatrixHandle {
        let mut handle = MatrixHandle::empty();
        let alloc_res = unsafe {
            matrix_alloc_or_reuse(&mut handle as *mut MatrixHandle, rows, columns)
        };
        if alloc_res != 0 {
            panic!("Failed to allocate empty matrix!");
        }
        return handle;
    }

    pub fn multiply(dst: &mut MatrixHandle, A: &MatrixHandle, B: &MatrixHandle) {
        let multiply_res = unsafe {
            matrix_multiply(A as *const MatrixHandle,
                            B as *const MatrixHandle,
                            dst as *mut MatrixHandle)
        };
        if multiply_res != 0 {
            panic!("Failed to multiply matrices: {}", multiply_res);
        }
    }

    pub fn rows(&self) -> usize {
        return self.rows;
    }

    pub fn columns(&self) -> usize {
        return self.columns;
    }

    pub fn entrywise_product(&self, rhs: &MatrixHandle) -> MatrixHandle {
        let mut result_handle = MatrixHandle::empty();
        let result = unsafe {
            matrix_entrywise_multiply(self as *const MatrixHandle,
                                      rhs as *const MatrixHandle,
                                      &mut result_handle as *mut MatrixHandle)
        };
        if result != 0 {
            panic!("Failed to entrywise-product matrices!");
        }
        return result_handle;
    }

    pub fn inplace_entrywise_product(&mut self, rhs: &MatrixHandle) {
        let add_assign_result = unsafe {
            matrix_inplace_entrywise_multiply(self as *mut MatrixHandle,
                                              rhs as *const MatrixHandle)
        };
        if add_assign_result != 0 {
            panic!("Failed to inplace entrywise product!");
        }
    }

    pub fn transpose(&self) -> MatrixHandle {
        let mut result_handle = MatrixHandle::empty();
        let result = unsafe {
            matrix_transpose(self as *const MatrixHandle,
                             &mut result_handle as *mut MatrixHandle)
        };
        if result != 0 {
            panic!("Failed to transpose matrices!");
        }
        return result_handle;
    }

    pub fn inplace_transpose(&mut self) {
        let result = unsafe {
            matrix_inplace_transpose(self as *mut MatrixHandle)
        };
        if result != 0 {
            panic!("Failed to inplace transpose!");
        }
    }

    pub fn inplace_scalar_multiply(&mut self, scalar: f32) {
        let result = unsafe {
            matrix_scalar_multiply(self as *const MatrixHandle,
                                   scalar,
                                   self as *mut MatrixHandle)
        };
        if result != 0 {
            panic!("Failed to inplace scalar multiply");
        }
    }

    pub fn add_constant_row(&self, padding: f32) -> MatrixHandle {
        let mut result_handle = MatrixHandle::empty();
        let result = unsafe {
            matrix_add_constant_row(padding,
                                    self as *const MatrixHandle,
                                    &mut result_handle as *mut MatrixHandle)
        };
        if result != 0 {
            panic!("Failed to add constant row!");
        }
        return result_handle;
    }

    pub fn inplace_add_constant_row(&mut self, padding: f32) {
        let result = unsafe {
            matrix_inplace_add_constant_row(padding, self as *mut MatrixHandle)
        };
        if result != 0 {
            panic!("Failed to inplace add constant row!");
        }
    }

    // Kind of hacky, assumes no side effects. Fine for now, and it's fast!
    pub fn inplace_remove_first_row(&mut self) {
        assert!(self.rows > 0, "Matrix must have > 0 row!");
        let result = unsafe {
            matrix_inplace_remove_first_row(self as *mut MatrixHandle)
        };
        if result != 0 {
            panic!("Failed to inplace remove first row!");
        }
    }

    pub fn dropout_elements(&self, rate: f32) -> MatrixHandle {
        let mut result_handle = MatrixHandle::empty();
        let result = unsafe {
            matrix_dropout_elements(rate,
                                    self as *const MatrixHandle,
                                    &mut result_handle as *mut MatrixHandle)
        };
        if result != 0 {
            panic!("Failed to transpose matrices!");
        }
        return result_handle;
    }

    pub fn dropout_rows(&self, rate: f32) -> MatrixHandle {
        let mut result_handle = MatrixHandle::empty();
        let result = unsafe {
            matrix_dropout_rows(rate,
                                self as *const MatrixHandle,
                                &mut result_handle as *mut MatrixHandle)
        };
        if result != 0 {
            panic!("Failed to transpose matrices!");
        }
        return result_handle;
    }
}

#[cfg(matrixlib)]
impl Drop for LibMatrixHandle {
    fn drop(&mut self) {
        if self.elements != ptr::null() {
            unsafe { matrix_free(self as *mut LibMatrixHandle) };
        }
    }
}

#[cfg(matrixlib)]
impl<'a> Add for &'a LibMatrixHandle {
    type Output = MatrixHandle;

    fn add(self, rhs: &LibMatrixHandle) -> LibMatrixHandle {
        let mut result_handle = LibMatrixHandle::empty();
        let add_result = unsafe {
            matrix_add(self as *const LibMatrixHandle,
                       rhs as *const LibMatrixHandle,
                       &mut result_handle as *mut LibMatrixHandle)
        };
        if add_result != 0 {
            panic!("Failed to add matrices!");
        }
        return result_handle;
    }
}

#[cfg(matrixlib)]
impl AddAssign for LibMatrixHandle {
    fn add_assign(&mut self, rhs: LibMatrixHandle) {
        let add_assign_result = unsafe {
            matrix_add_assign(self as *mut LibMatrixHandle,
                              &rhs as *const LibMatrixHandle)
        };
        if add_assign_result != 0 {
            panic!("Failed to add assign matrices!");
        }
    }
}

#[cfg(matrixlib)]
impl<'a> AddAssign<&'a LibMatrixHandle> for LibMatrixHandle {
    fn add_assign(&mut self, rhs: &LibMatrixHandle) {
        let add_assign_result = unsafe {
            matrix_add_assign(self as *mut LibMatrixHandle,
                              rhs as *const LibMatrixHandle)
        };
        if add_assign_result != 0 {
            panic!("Failed to add assign matrices!");
        }
    }
}

#[cfg(matrixlib)]
impl<'a> Sub for &'a LibMatrixHandle {
    type Output = LibMatrixHandle;

    fn sub(self, rhs: &LibMatrixHandle) -> LibMatrixHandle {
        let mut result_handle = LibMatrixHandle::empty();
        let add_result = unsafe {
            matrix_sub(self as *const LibMatrixHandle,
                       rhs as *const LibMatrixHandle,
                       &mut result_handle as *mut LibMatrixHandle)
        };
        if add_result != 0 {
            panic!("Failed to subtract matrices!");
        }
        return result_handle;
    }
}

#[cfg(matrixlib)]
impl<'a> SubAssign<&'a LibMatrixHandle> for LibMatrixHandle {
    fn sub_assign(&mut self, rhs: &LibMatrixHandle) {
        let add_assign_result = unsafe {
            matrix_sub_assign(self as *mut LibMatrixHandle,
                              rhs as *const LibMatrixHandle)
        };
        if add_assign_result != 0 {
            panic!("Failed to add assign matrices!");
        }
    }
}

#[cfg(matrixlib)]
impl<'a> Mul for &'a LibMatrixHandle {
    type Output = LibMatrixHandle;

    fn mul(self, rhs: &LibMatrixHandle) -> LibMatrixHandle {
        let mut result_handle = LibMatrixHandle::empty();
        let add_result = unsafe {
            matrix_multiply(self as *const LibMatrixHandle,
                            rhs as *const LibMatrixHandle,
                            &mut result_handle as *mut LibMatrixHandle)
        };
        if add_result != 0 {
            panic!("Failed to multiply matrices!");
        }
        return result_handle;
    }
}

#[cfg(matrixlib)]
impl<'a> Mul<f32> for &'a LibMatrixHandle {
    type Output = LibMatrixHandle;

    fn mul(self, scalar: f32) -> LibMatrixHandle {
        let mut result_handle = LibMatrixHandle::empty();
        let add_result = unsafe {
            matrix_scalar_multiply(self as *const LibMatrixHandle,
                                   scalar,
                                   &mut result_handle as *mut LibMatrixHandle)
        };
        if add_result != 0 {
            panic!("Failed to scalar multiply matrices!");
        }
        return result_handle;
    }
}

#[cfg(matrixlib)]
impl<'a> Mul<&'a LibMatrixHandle> for f32 {
    type Output = LibMatrixHandle;

    fn mul(self, scalar: &LibMatrixHandle) -> LibMatrixHandle {
        return scalar * self;
    }
}

#[cfg(matrixlib)]
impl Clone for LibMatrixHandle {
    fn clone(&self) -> LibMatrixHandle {
        let mut result_handle = LibMatrixHandle::empty();
        let copy_result = unsafe {
            matrix_copy(self as *const LibMatrixHandle,
                        &mut result_handle as *mut LibMatrixHandle)
        };
        if copy_result != 0 {
            panic!("Failed to add matrices!");
        }
        return result_handle;
    }
}

#[cfg(matrixlib)]
impl Serialize for LibMatrixHandle {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer
    {
        let matrix = MatrixHandle::to_matrix(self);
        matrix.serialize(serializer)
    }
}

#[cfg(matrixlib)]
impl<'de> Deserialize<'de> for LibMatrixHandle {
    fn deserialize<D>(deserializer: D) -> Result<LibMatrixHandle, D::Error>
        where D: Deserializer<'de>
    {
        let deserialize_result = Matrix::deserialize(deserializer);
        deserialize_result.map(|matrix| MatrixHandle::from_matrix(matrix))
    }
}