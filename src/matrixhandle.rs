use serde::{ Deserializer, Deserialize, Serializer, Serialize };
use matrix::Matrix;
use std::ops::{Add, AddAssign, Sub, Mul};
use std::ptr;

#[repr(C)]
pub struct MatrixHandle {
    rows: usize,
    columns: usize,
    elements: *const f32
}

unsafe impl Send for MatrixHandle { }

#[link(name = "matrix", kind = "static")]
extern {

    fn matrix_synchronize(only_current_thread: u8);

    fn matrix_alloc(rows: libc::size_t,
                    columns: libc::size_t,
                    elements: *const libc::c_float,
                    handle: *mut MatrixHandle) -> libc::c_int;

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

    fn matrix_entrywise_multiply(handle_a: *const MatrixHandle,
                                 handle_b: *const MatrixHandle,
                                 handle_result: *mut MatrixHandle) -> libc::c_int;

    fn matrix_scalar_multiply(handle_a: *const MatrixHandle,
                              scalar: libc::c_float,
                              handle_result: *mut MatrixHandle) -> libc::c_int;

    fn matrix_multiply(handle_a: *const MatrixHandle,
                       handle_b: *const MatrixHandle,
                       handle_result: *mut MatrixHandle) -> libc::c_int;
    
    fn matrix_transpose(handle_a: *const MatrixHandle,
                        handle_result: *mut MatrixHandle) -> libc::c_int;

    fn matrix_add_constant_row(padding: f32,
                               handle_a: *const MatrixHandle,
                               handle_result: *mut MatrixHandle) -> libc::c_int;

    fn matrix_dropout_elements(rate: f32,
                               handle_a: *const MatrixHandle,
                               handle_result: *mut MatrixHandle) -> libc::c_int;

    fn matrix_dropout_rows(rate: f32,
                           handle_a: *const MatrixHandle,
                           handle_result: *mut MatrixHandle) -> libc::c_int;
    
    fn matrix_copy(handle_a: *const MatrixHandle,
                   handle_result: *mut MatrixHandle) -> libc::c_int;
}

impl MatrixHandle {
    pub fn empty() -> MatrixHandle {
        MatrixHandle {
            rows: 0,
            columns: 0,
            elements: ptr::null()
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

    pub fn from_matrix(matrix: Matrix<f32>) -> MatrixHandle {
        let mut handle: MatrixHandle = MatrixHandle::empty();

        let alloc_result = unsafe {
            matrix_alloc(matrix.rows(),
                         matrix.columns(),
                         matrix.raw_values().as_ptr(),
                         &mut handle as *mut MatrixHandle)
        };
        if alloc_result != 0 {
            panic!("Failed to create MatrixHandle");
        }

        return handle;
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

    pub fn add_constant_row(&self, padding: f32) -> MatrixHandle {
        let mut result_handle = MatrixHandle::empty();
        let result = unsafe {
            matrix_add_constant_row(padding,
                                    self as *const MatrixHandle,
                                    &mut result_handle as *mut MatrixHandle)
        };
        if result != 0 {
            panic!("Failed to transpose matrices!");
        }
        return result_handle;
    }

    // Kind of hacky, assumes no side effects. Fine for now, and it's fast!
    pub fn remove_first_row(&self) -> MatrixHandle {
        assert!(self.rows > 1, "Matrix must have > 1 row!");

        MatrixHandle {
            rows: self.rows - 1,
            columns: self.columns,
            elements: unsafe { self.elements.offset(self.columns as isize) }
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

impl Drop for MatrixHandle {
    fn drop(&mut self) {
        if self.elements != ptr::null() {
            unsafe { matrix_free(self as *mut MatrixHandle) };
        }
    }
}

impl<'a> Add for &'a MatrixHandle {
    type Output = MatrixHandle;

    fn add(self, rhs: &MatrixHandle) -> MatrixHandle {
        let mut result_handle = MatrixHandle::empty();
        let add_result = unsafe {
            matrix_add(self as *const MatrixHandle,
                       rhs as *const MatrixHandle,
                       &mut result_handle as *mut MatrixHandle)
        };
        if add_result != 0 {
            panic!("Failed to add matrices!");
        }
        return result_handle;
    }
}

impl AddAssign for MatrixHandle {
    fn add_assign(&mut self, rhs: MatrixHandle) {
        let add_assign_result = unsafe {
            matrix_add_assign(self as *mut MatrixHandle,
                              &rhs as *const MatrixHandle)
        };
        if add_assign_result != 0 {
            panic!("Failed to add assign matrices!");
        }
    }
}

impl<'a> Sub for &'a MatrixHandle {
    type Output = MatrixHandle;

    fn sub(self, rhs: &MatrixHandle) -> MatrixHandle {
        let mut result_handle = MatrixHandle::empty();
        let add_result = unsafe {
            matrix_sub(self as *const MatrixHandle,
                       rhs as *const MatrixHandle,
                       &mut result_handle as *mut MatrixHandle)
        };
        if add_result != 0 {
            panic!("Failed to add matrices!");
        }
        return result_handle;
    }
}

impl<'a> Mul for &'a MatrixHandle {
    type Output = MatrixHandle;

    fn mul(self, rhs: &MatrixHandle) -> MatrixHandle {
        let mut result_handle = MatrixHandle::empty();
        let add_result = unsafe {
            matrix_multiply(self as *const MatrixHandle,
                            rhs as *const MatrixHandle,
                            &mut result_handle as *mut MatrixHandle)
        };
        if add_result != 0 {
            panic!("Failed to add matrices!");
        }
        return result_handle;
    }
}

impl<'a> Mul<f32> for &'a MatrixHandle {
    type Output = MatrixHandle;

    fn mul(self, scalar: f32) -> MatrixHandle {
        let mut result_handle = MatrixHandle::empty();
        let add_result = unsafe {
            matrix_scalar_multiply(self as *const MatrixHandle,
                                   scalar,
                                   &mut result_handle as *mut MatrixHandle)
        };
        if add_result != 0 {
            panic!("Failed to add matrices!");
        }
        return result_handle;
    }
}

impl<'a> Mul<&'a MatrixHandle> for f32 {
    type Output = MatrixHandle;

    fn mul(self, scalar: &MatrixHandle) -> MatrixHandle {
        return scalar * self;
    }
}

impl Clone for MatrixHandle {
    fn clone(&self) -> MatrixHandle {
        let mut result_handle = MatrixHandle::empty();
        let copy_result = unsafe {
            matrix_copy(self as *const MatrixHandle,
                        &mut result_handle as *mut MatrixHandle)
        };
        if copy_result != 0 {
            panic!("Failed to add matrices!");
        }
        return result_handle;
    }
}

impl Serialize for MatrixHandle {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer
    {
        let matrix = MatrixHandle::to_matrix(self);
        matrix.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for MatrixHandle {
    fn deserialize<D>(deserializer: D) -> Result<MatrixHandle, D::Error>
        where D: Deserializer<'de>
    {
        let deserialize_result = Matrix::deserialize(deserializer);
        deserialize_result.map(|matrix| MatrixHandle::from_matrix(matrix))
    }
}
