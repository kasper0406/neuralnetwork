use metal::*;
use objc_foundation::{INSString, INSArray};

use matrix::Matrix;
use std::ops::{Add, Mul};

lazy_static! {
    static ref METAL_INSTANCE: MetalInstance = MetalInstance::new();
}

pub struct MetalMatrixHandle {
    rows: u64,
    columns: u64,
    data: Buffer
}

impl MetalMatrixHandle {
    fn from_matrix(matrix: &Matrix<f32>) -> MetalMatrixHandle {
        let mut buf = Vec::with_capacity(matrix.rows() * matrix.columns());
        for row in 0..matrix.rows() {
            for col in 0..matrix.columns() {
                buf.push(matrix[(row, col)]);
            }
        }

        let data_buffer = METAL_INSTANCE.copy_to_gpu(&buf);

        return MetalMatrixHandle {
            rows: matrix.rows() as u64,
            columns: matrix.columns() as u64,
            data: data_buffer
        };
    }

    fn to_matrix(&self) -> Matrix<f32> {
        let buf = METAL_INSTANCE.copy_from_gpu(&self.data);
        return Matrix::new(self.rows(), self.columns(), &|row, col| buf[row * self.columns() + col])
    }

    fn rows(&self) -> usize {
        return self.rows as usize;
    }

    fn columns(&self) -> usize {
        return self.columns as usize;
    }

    fn encode_to_metal_buffer(&self,
                              command_encoder: &ComputeCommandEncoderRef,
                              kernel: &Function, index: u64) -> Buffer {
        let arg_encoder = kernel.new_argument_encoder(index);
        let matrix_buffer = METAL_INSTANCE.device.new_buffer(
            arg_encoder.encoded_length(),
            MTLResourceOptions::empty()
        );
        arg_encoder.set_argument_buffer(&matrix_buffer, 0);

        let data_ptr = arg_encoder.constant_data(0) as *mut u64;
        unsafe { std::ptr::write(data_ptr, self.rows) };

        let data_ptr = arg_encoder.constant_data(1) as *mut u64;
        unsafe { std::ptr::write(data_ptr, self.columns) };

        arg_encoder.set_buffer(&self.data, 0, 2);

        // Signal to the command encoder that the matrix data will be used in the computation
        command_encoder.use_resource(&self.data, MTLResourceUsage::Read);

        return matrix_buffer;
    }
}

impl<'a> Add<&'a MetalMatrixHandle> for &'a MetalMatrixHandle {
    type Output = MetalMatrixHandle;

    fn add(self, rhs: &'a MetalMatrixHandle) -> MetalMatrixHandle {
        assert!(self.rows() == rhs.rows(), "Number of rows must match");
        assert!(self.columns() == rhs.columns(), "Number of columns must match");

        let count = self.rows() * self.columns();
        let res_buffer = METAL_INSTANCE.device.new_buffer(
            (count * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModePrivate
        );

        let command_buffer = METAL_INSTANCE.command_queue.new_command_buffer();
        let command_encoder = command_buffer.new_compute_command_encoder();
        
        command_encoder.set_compute_pipeline_state(&METAL_INSTANCE.sum_pipeline_state);
        command_encoder.set_buffers(0, &[ Some(&self.data), Some(&rhs.data), Some(&res_buffer) ], &[ 0, 0, 0 ]);

        let threads_per_threadgroup = MTLSize {
            width: METAL_INSTANCE.sum_pipeline_state.thread_execution_width(), // Consider if using maxTotalThreadsPerThreadgroup is better here
            height: 1,
            depth: 1
        };
        let threads_per_grid = MTLSize {
            width: count as u64,
            height: 1,
            depth: 1
        };

        unsafe {
            msg_send![command_encoder,
                dispatchThreads: threads_per_grid
                threadsPerThreadgroup: threads_per_threadgroup
            ]
        }

        command_encoder.end_encoding();
        command_buffer.commit();

        return MetalMatrixHandle {
            rows: self.rows,
            columns: self.columns,
            data: res_buffer
        }
    }
}

impl<'a> Mul<&'a MetalMatrixHandle> for &'a MetalMatrixHandle {
    type Output = MetalMatrixHandle;

    fn mul(self, rhs: &'a MetalMatrixHandle) -> MetalMatrixHandle {
        assert!(self.columns() == rhs.rows(), "LHS columns must match RHS rows");

        let res_rows = self.rows();
        let res_columns = rhs.columns();
        let res_buffer = METAL_INSTANCE.device.new_buffer(
            (res_rows * res_columns * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModePrivate
        );

        let command_buffer = METAL_INSTANCE.command_queue.new_command_buffer();
        let command_encoder = command_buffer.new_compute_command_encoder();

        let self_buffer = self.encode_to_metal_buffer(command_encoder, &METAL_INSTANCE.mul_kernel, 0);
        let rhs_buffer = rhs.encode_to_metal_buffer(command_encoder, &METAL_INSTANCE.mul_kernel, 0);

        command_encoder.set_compute_pipeline_state(&METAL_INSTANCE.mul_pipeline_state);
        command_encoder.set_buffer(0, Some(&self_buffer), 0);
        command_encoder.set_buffer(1, Some(&rhs_buffer), 0);
        command_encoder.set_buffer(2, Some(&res_buffer), 0);

        let threads_per_threadgroup = MTLSize {
            width: METAL_INSTANCE.sum_pipeline_state.thread_execution_width(),
            height: METAL_INSTANCE.sum_pipeline_state.max_total_threads_per_group() / METAL_INSTANCE.sum_pipeline_state.thread_execution_width(),
            depth: 1
        };
        let threads_per_grid = MTLSize {
            width: self.rows,
            height: rhs.columns,
            depth: 1
        };

        unsafe {
            msg_send![command_encoder,
                dispatchThreads: threads_per_grid
                threadsPerThreadgroup: threads_per_threadgroup
            ]
        }

        command_encoder.end_encoding();
        command_buffer.commit();

        return MetalMatrixHandle {
            rows: res_rows as u64,
            columns: res_columns as u64,
            data: res_buffer
        }
    }
}

struct MetalInstance {
    device: Device,
    command_queue: CommandQueue,

    sum_pipeline_state: ComputePipelineState,

    mul_kernel: Function,
    mul_pipeline_state: ComputePipelineState
}

unsafe impl Sync for MetalInstance { }

impl MetalInstance {
    pub fn new() -> MetalInstance {
        let device = Device::system_default();
        println!("Running matrix computations on {}", device.name());

        let command_queue = device.new_command_queue();

        let metal_matrix_lib = device.new_library_with_file("src/metalmatrix.metallib").unwrap();

        println!("Found the following functions in the Metal library:");
        let function_names = metal_matrix_lib.function_names().to_vec();
        for name in function_names {
            println!("{}", name.as_str());
        }

        let sum_kernel = metal_matrix_lib.get_function("add_arrays", None).unwrap();
        let sum_pipeline_state = device.new_compute_pipeline_state_with_function(&sum_kernel).unwrap();

        let mul_kernel = metal_matrix_lib.get_function("multiply", None).unwrap();
        let mul_pipeline_state = device.new_compute_pipeline_state_with_function(&mul_kernel).unwrap();

        return MetalInstance {
            device, command_queue, sum_pipeline_state, mul_kernel, mul_pipeline_state
        }
    }

    pub fn copy_to_gpu(&self, data: &[f32]) -> Buffer {
        let command_buffer = self.command_queue.new_command_buffer();
        let size = (data.len() * std::mem::size_of::<f32>()) as u64;

        let mut buffer = self.device.new_buffer(
            size,
            MTLResourceOptions::StorageModePrivate
        );

        let tmp = self.device.new_buffer_with_data(
            unsafe { std::mem::transmute(data.as_ptr()) },
            size,
            MTLResourceOptions::CPUCacheModeDefaultCache
        );

        let blit_encoder = command_buffer.new_blit_command_encoder();
        blit_encoder.copy_from_buffer(&tmp, 0, &mut buffer, 0, size);
        blit_encoder.end_encoding();

        command_buffer.commit();

        return buffer;
    }

    pub fn copy_from_gpu(&self, buffer: &Buffer) -> Vec<f32> {
        let command_buffer = self.command_queue.new_command_buffer();

        let size = buffer.length();
        assert!(size as usize % std::mem::size_of::<f32>() == 0);
        let count = size as usize / std::mem::size_of::<f32>();

        let mut tmp = self.device.new_buffer(size, MTLResourceOptions::StorageModeShared);

        let blit_encoder = command_buffer.new_blit_command_encoder();
        blit_encoder.copy_from_buffer(&buffer, 0, &mut tmp, 0, size);
        blit_encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let raw_ptr: *const f32 = unsafe { std::mem::transmute(tmp.contents()) };
        let mut dst = Vec::with_capacity(count);
        for i in 0..count {
            dst.push(unsafe { raw_ptr.offset(i as isize).read() });
        }
        return dst;
    }
}

pub fn test() {
    let A = Matrix::new(50, 50, &|row, col| if row == col { 1 } else { 0 } as f32);
    let B = Matrix::new(50, 50, &|row, col| if row == col { 1 } else { 0 } as f32);

    let handleA = MetalMatrixHandle::from_matrix(&A);
    let handleB = MetalMatrixHandle::from_matrix(&B);

    let handleC = &handleA * &handleB;

    let C = MetalMatrixHandle::to_matrix(&(&(&handleC + &handleC) * &(&handleA + &handleB)));

    // let C = &A * &B;

    println!("Result of multiplication:");
    println!("{}", C);

    /*
    let handleC = &handleA + &handleB;

    let C = MetalMatrixHandle::to_matrix(&handleC);
    println!("Result of addition:");
    println!("{}", C); */

    // let B = Matrix::new(5, 5, &|row, col| row as f32 / col as f32);
}
