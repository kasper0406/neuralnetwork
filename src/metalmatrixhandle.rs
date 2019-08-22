use metal::*;
use objc_foundation::{INSString, INSArray};

use matrixhandle::MatrixHandle;
use matrix::Matrix;
use std::ops::{Add, AddAssign, Sub, SubAssign, Mul};
use serde::{ Deserializer, Deserialize, Serializer, Serialize };

use cocoa::foundation::NSAutoreleasePool;

lazy_static! {
    static ref METAL_INSTANCE: MetalInstance = MetalInstance::new();
}

pub enum ElementFunction {
    Sigmoid, SigmoidDerivative
}

pub struct MetalMatrixHandle {
    rows: u64,
    columns: u64,
    data: Buffer
}

impl MetalMatrixHandle {
    fn encode_to_metal_buffer<'a>(&self,
                                  command_encoder: &'a ComputeCommandEncoderRef,
                                  kernel: &Function,
                                  index: u64) -> Buffer {
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

    pub fn apply_function(&self, function: ElementFunction) -> MetalMatrixHandle {
        let kernel = match function {
            ElementFunction::Sigmoid => &METAL_INSTANCE.sigmoid_kernel,
            ElementFunction::SigmoidDerivative => &METAL_INSTANCE.sigmoid_derivative_kernel,
        };

        return METAL_INSTANCE.execute_kernel_with_result_alloc(
            kernel,
            &[BufferDescription::MetalBuffer(&self.data)],
            (1, self.rows() * self.columns()),
            (self.rows(), self.columns())
        );
    }
}

impl MatrixHandle for MetalMatrixHandle {

    fn of_size(rows: usize, columns: usize) -> MetalMatrixHandle {
        panic!("Method 'of_size' not yet implemented!");
    }

    fn copy_from_matrix(dst: &mut MetalMatrixHandle, matrix: Matrix<f32>) {
        panic!("Method 'copy_from_matrix' not yet implemented!");
    }

    fn copy(destination: &mut Self, source: &Self) {
        panic!("Method 'copy' not yet implemented!");
    }

    fn inplace_entrywise_product(&mut self, rhs: &Self) {
        panic!("Method 'inplace_entrywise_product' not yet implemented!");
    }
    fn inplace_add_constant_row(&mut self, value: f32) {
        panic!("Method 'inplace_add_constant_row' not yet implemented!");
    }
    fn inplace_remove_first_row(&mut self) {
        panic!("Method 'inplace_remove_first_row' not yet implemented!");
    }
    fn inplace_transpose(&mut self) {
        panic!("Method 'inplace_transpose' not yet implemented!");
    }
    fn inplace_scalar_multiply(&mut self, scalar: f32) {
        panic!("Method 'inplace_scalar_multiply' not yet implemented!");
    }

    fn multiply(lhs: &Self, rhs: &Self, dst: &mut Self) {
        panic!("Method 'multiply' not yet implemented!");
    }

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

    fn entrywise_product(&self, rhs: &Self) -> Self {
        return METAL_INSTANCE.execute_kernel_with_result_alloc(
            &METAL_INSTANCE.entrywise_product_kernel,
            &[BufferDescription::MetalBuffer(&self.data), BufferDescription::MetalBuffer(&rhs.data)],
            (1, self.rows() * self.columns()),
            (self.rows(), self.columns())
        );
    }

    fn transpose(&self) -> Self {
        return METAL_INSTANCE.execute_kernel_with_result_alloc(
            &METAL_INSTANCE.transpose_kernel,
            &[BufferDescription::Matrix(&self)],
            (self.rows(), self.columns()),
            (self.columns(), self.rows())
        );
    }

    fn add_constant_row(&self, value: f32) -> Self {
        let new_count = (self.rows() + 1) * self.columns();
        let old_size = (self.rows() * self.columns() * std::mem::size_of::<f32>()) as u64;
        let new_size = (new_count * std::mem::size_of::<f32>()) as u64;

        let command_buffer = METAL_INSTANCE.command_queue.new_command_buffer();

        let mut buffer = METAL_INSTANCE.device.new_buffer(
            new_size,
            MTLResourceOptions::StorageModePrivate
        );

        let blit_encoder = command_buffer.new_blit_command_encoder();
        let bytes_per_row = self.columns * std::mem::size_of::<f32>() as u64;

        let data = vec![value; self.columns()];
        let buffer_with_first_row_values = METAL_INSTANCE.device.new_buffer_with_data(
            unsafe { std::mem::transmute(data.as_ptr()) },
            bytes_per_row,
            MTLResourceOptions::CPUCacheModeDefaultCache
        );
        blit_encoder.copy_from_buffer(&buffer_with_first_row_values, 0, &mut buffer, 0, bytes_per_row);
        blit_encoder.copy_from_buffer(&self.data, 0, &mut buffer, bytes_per_row, old_size);
        blit_encoder.end_encoding();

        command_buffer.commit();

        return MetalMatrixHandle {
            rows: self.rows + 1,
            columns: self.columns,
            data: buffer
        };
    }

    fn remove_first_row(&self) -> Self {
        assert!(self.rows() > 1, "The matrix must have at >1 row to remove");

        let new_count = (self.rows() - 1) * self.columns();
        let new_size = (new_count * std::mem::size_of::<f32>()) as u64;

        let command_buffer = METAL_INSTANCE.command_queue.new_command_buffer();

        let mut buffer = METAL_INSTANCE.device.new_buffer(
            new_size,
            MTLResourceOptions::StorageModePrivate
        );

        let blit_encoder = command_buffer.new_blit_command_encoder();
        blit_encoder.copy_from_buffer(&self.data, self.columns * std::mem::size_of::<f32>() as u64, &mut buffer, 0, new_size);
        blit_encoder.end_encoding();

        command_buffer.commit();

        return MetalMatrixHandle {
            rows: self.rows - 1,
            columns: self.columns,
            data: buffer
        };
    }

    fn dropout_elements(&self, rate: f32) -> Self {
        panic!("Method 'dropout_elements' not yet implemented!");
    }

    fn dropout_rows(&self, rate: f32) -> Self {
        panic!("Method 'dropout_rows' not yet implemented!");
    }
}

impl Add<MetalMatrixHandle> for MetalMatrixHandle {
    type Output = MetalMatrixHandle;

    fn add(self, rhs: MetalMatrixHandle) -> MetalMatrixHandle {
        return &self + &rhs;
    }
}
impl<'a> Add<&'a MetalMatrixHandle> for &'a MetalMatrixHandle {
    type Output = MetalMatrixHandle;

    fn add(self, rhs: &'a MetalMatrixHandle) -> MetalMatrixHandle {
        return METAL_INSTANCE.execute_kernel_with_result_alloc(
            &METAL_INSTANCE.add_kernel,
            &[BufferDescription::MetalBuffer(&self.data), BufferDescription::MetalBuffer(&rhs.data)],
            (1, self.rows() * self.columns()),
            (self.rows(), self.columns())
        );
    }
}

impl<'a> AddAssign for MetalMatrixHandle {
    fn add_assign(&mut self, rhs: MetalMatrixHandle) {
        return METAL_INSTANCE.execute_kernel(
            &METAL_INSTANCE.add_kernel,
            &[BufferDescription::MetalBuffer(&self.data), BufferDescription::MetalBuffer(&rhs.data)],
            &self.data,
            (1, self.rows() * self.columns())
        );
    }
}

impl Sub<MetalMatrixHandle> for MetalMatrixHandle {
    type Output = MetalMatrixHandle;

    fn sub(self, rhs: MetalMatrixHandle) -> MetalMatrixHandle {
        return &self - &rhs;
    }
}
impl<'a> Sub<&'a MetalMatrixHandle> for &'a MetalMatrixHandle {
    type Output = MetalMatrixHandle;

    fn sub(self, rhs: &'a MetalMatrixHandle) -> MetalMatrixHandle {
        return METAL_INSTANCE.execute_kernel_with_result_alloc(
            &METAL_INSTANCE.sub_kernel,
            &[BufferDescription::MetalBuffer(&self.data), BufferDescription::MetalBuffer(&rhs.data)],
            (1, self.rows() * self.columns()),
            (self.rows(), self.columns())
        );
    }
}

impl<'a> SubAssign for MetalMatrixHandle {
    fn sub_assign(&mut self, rhs: MetalMatrixHandle) {
        return METAL_INSTANCE.execute_kernel(
            &METAL_INSTANCE.sub_kernel,
            &[BufferDescription::MetalBuffer(&self.data), BufferDescription::MetalBuffer(&rhs.data)],
            &self.data,
            (1, self.rows() * self.columns())
        );
    }
}

impl Mul for MetalMatrixHandle {
    type Output = MetalMatrixHandle;

    fn mul(self, rhs: MetalMatrixHandle) -> MetalMatrixHandle {
        return &self * &rhs;
    }
}
impl<'a> Mul<&'a MetalMatrixHandle> for &'a MetalMatrixHandle {
    type Output = MetalMatrixHandle;

    fn mul(self, rhs: &'a MetalMatrixHandle) -> MetalMatrixHandle {
        assert!(self.columns() == rhs.rows(), "LHS columns must match RHS rows");

        return METAL_INSTANCE.execute_kernel_with_result_alloc(
            &METAL_INSTANCE.mul_kernel,
            &[BufferDescription::Matrix(&self), BufferDescription::Matrix(&rhs)],
            (self.rows(), rhs.columns()),
            (self.rows(), rhs.columns())
        );
    }
}

impl Mul<f32> for MetalMatrixHandle {
    type Output = MetalMatrixHandle;
    fn mul(self, scalar: f32) -> MetalMatrixHandle {
        return &self * scalar;
    }
}
impl<'a> Mul<f32> for &'a MetalMatrixHandle {
    type Output = MetalMatrixHandle;

    fn mul(self, scalar: f32) -> MetalMatrixHandle {
        let data = [scalar; 1];
        let scalar_buffer = METAL_INSTANCE.device.new_buffer_with_data(
            unsafe { std::mem::transmute(data.as_ptr()) },
            std::mem::size_of::<f32>() as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache
        );

        return METAL_INSTANCE.execute_kernel_with_result_alloc(
            &METAL_INSTANCE.scalar_multiply_kernel,
            &[BufferDescription::MetalBuffer(&scalar_buffer), BufferDescription::MetalBuffer(&self.data)],
            (1, self.rows() * self.columns()),
            (self.rows(), self.columns())
        );
    }
}
impl Mul<&MetalMatrixHandle> for f32 {
    type Output = MetalMatrixHandle;
    fn mul(self, matrix: &MetalMatrixHandle) -> MetalMatrixHandle {
        return matrix * self;
    }
}

impl Clone for MetalMatrixHandle {
    fn clone(&self) -> MetalMatrixHandle {
        let count = self.rows() * self.columns();
        let size = (count * std::mem::size_of::<f32>()) as u64;

        let command_buffer = METAL_INSTANCE.command_queue.new_command_buffer();

        let mut buffer = METAL_INSTANCE.device.new_buffer(
            size,
            MTLResourceOptions::StorageModePrivate
        );
        unsafe { msg_send![buffer, retain] };

        let blit_encoder = command_buffer.new_blit_command_encoder();
        blit_encoder.copy_from_buffer(&self.data, 0, &mut buffer, 0, size);
        blit_encoder.end_encoding();

        command_buffer.commit();

        return MetalMatrixHandle {
            rows: self.rows,
            columns: self.columns,
            data: buffer
        };
    }
}

enum BufferDescription<'a> {
    MetalBuffer(&'a Buffer),
    Matrix(&'a MetalMatrixHandle),
}

struct KernelDescriptor {
    function: Function,
    pipeline_state: ComputePipelineState,
}

struct MetalInstance {
    device: Device,
    command_queue: CommandQueue,

    add_kernel: KernelDescriptor,
    sub_kernel: KernelDescriptor,
    entrywise_product_kernel: KernelDescriptor,
    transpose_kernel: KernelDescriptor,
    mul_kernel: KernelDescriptor,
    scalar_multiply_kernel: KernelDescriptor,

    sigmoid_kernel: KernelDescriptor,
    sigmoid_derivative_kernel: KernelDescriptor,
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

        let register_kernel = |name| {
            let function = metal_matrix_lib.get_function(name, None).unwrap();
            let pipeline_state = device.new_compute_pipeline_state_with_function(&function).unwrap();
            return KernelDescriptor { function, pipeline_state };
        };

        return MetalInstance {
            command_queue,
            add_kernel: register_kernel("add"),
            sub_kernel: register_kernel("sub"),
            entrywise_product_kernel: register_kernel("entrywise_product"),
            transpose_kernel: register_kernel("transpose"),
            mul_kernel: register_kernel("mul"),
            scalar_multiply_kernel: register_kernel("scalar_multiply"),

            sigmoid_kernel: register_kernel("sigmoid"),
            sigmoid_derivative_kernel: register_kernel("sigmoid_derivative"),

            device
        }
    }

    pub fn copy_to_gpu(&self, data: &[f32]) -> Buffer {
        let command_buffer = self.command_queue.new_command_buffer();
        let size = (data.len() * std::mem::size_of::<f32>()) as u64;

        let mut buffer = self.device.new_buffer(
            size,
            MTLResourceOptions::StorageModePrivate
        );
        // unsafe { msg_send![buffer, retain] };

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

    fn execute_kernel(&self,
                      kernel: &KernelDescriptor,
                      buffers: &[BufferDescription],
                      result_buffer: &Buffer,
                      grid_size: (usize, usize)) {
        let command_buffer = self.command_queue.new_command_buffer();
        let command_encoder = command_buffer.new_compute_command_encoder();

        let pipeline_state = &kernel.pipeline_state;
        command_encoder.set_compute_pipeline_state(&pipeline_state);

        let mut _lifetime_buffers = Vec::new();
        for (i, buffer_description) in buffers.iter().enumerate() {
            let buffer = match buffer_description {
                BufferDescription::MetalBuffer(buffer) => buffer,
                BufferDescription::Matrix(handle) => {
                    let buffer = handle.encode_to_metal_buffer(command_encoder, &kernel.function, i as u64);

                    // Make lifetime checker happy
                    _lifetime_buffers.push(buffer);
                    _lifetime_buffers.last().unwrap()
                }
            };

            command_encoder.set_buffer(i as u64, Some(buffer), 0);
        }
        command_encoder.set_buffer(buffers.len() as u64, Some(result_buffer), 0);

        let metal_grid_size = MTLSize {
            width: grid_size.1 as u64,
            height: grid_size.0 as u64,
            depth: 1
        };

        let threads_per_threadgroup = MTLSize {
            width: u64::min(metal_grid_size.width, pipeline_state.thread_execution_width()),
            height: u64::min(metal_grid_size.height, pipeline_state.max_total_threads_per_group() / pipeline_state.thread_execution_width()),
            depth: 1
        };

        unsafe {
            msg_send![command_encoder,
                dispatchThreads: metal_grid_size
                threadsPerThreadgroup: threads_per_threadgroup
            ]
        }

        command_encoder.end_encoding();
        command_buffer.commit();
    }

    fn execute_kernel_with_result_alloc(&self,
                                        pipeline_state: &KernelDescriptor,
                                        buffers: &[BufferDescription],
                                        grid_size: (usize, usize),
                                        result_size: (usize, usize)) -> MetalMatrixHandle
    {
        let count = result_size.0 * result_size.1;
        let res_buffer = self.device.new_buffer(
            (count * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModePrivate
        );

        self.execute_kernel(pipeline_state, &buffers, &res_buffer, grid_size);

        return MetalMatrixHandle {
            rows: result_size.0 as u64,
            columns: result_size.1 as u64,
            data: res_buffer
        }
    }
}

unsafe impl Send for MetalMatrixHandle { }

impl Drop for MetalMatrixHandle {
    fn drop(&mut self) {
        // println!("Dropping {}x{} matrix", self.rows(), self.columns());
        unsafe {
            // msg_send![ self.data, release ];
        }
        // println!("Finished dropping");
    }
}

impl Serialize for MetalMatrixHandle {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer
    {
        panic!("Serialize not implemented!");
    }
}

impl<'de> Deserialize<'de> for MetalMatrixHandle {
    fn deserialize<D>(deserializer: D) -> Result<MetalMatrixHandle, D::Error>
        where D: Deserializer<'de>
    {
        panic!("Deserialize not implemented!");
    }
}

pub fn test() {

    // let pool = unsafe { NSAutoreleasePool::new(cocoa::base::nil) };

    /*
    let A = Matrix::new(10, 3, &|row, col| (row * 3 + col) as f32);
    println!("{}", A);
    println!("");

    let handleA = MetalMatrixHandle::from_matrix(&A);
    let handleB = handleA.transpose();
    let R = MetalMatrixHandle::to_matrix(&handleB); */

    let A = Matrix::new(100, 100, &|row, col| if row == col { 1 } else { 0 } as f32);
    // let B = Matrix::new(50, 50, &|row, col| if row == col { 1 } else { 0 } as f32);

    println!("Creating handles");

    let mut handleA = MetalMatrixHandle::from_matrix(&A);
    for _ in 0..10000 {
        handleA = MetalMatrixHandle::from_matrix(&A);
    }
    // let handleB = MetalMatrixHandle::from_matrix(&B);

    /*
    println!("Doing operations");
    let handleC = &handleA + &handleB;
    let handleD = &handleC * &handleC;
    let handleE = &handleD - &handleA;

    println!("Entrywise product and transpose");
    let handleF = handleD.entrywise_product(&handleE);
    let handleG = handleF.transpose();

    println!("Add and remove rows");
    let handleH = handleG.remove_first_row();
    let handleI = handleH.add_constant_row(1.0);

    println!("Create another matrix");
    let J = Matrix::new(50, 50, &|row, col| 1_f32);
    let mut handleJ = MetalMatrixHandle::from_matrix(&J);

    println!("Inplace operations");
    handleJ += handleI;
    handleJ -= handleA;
    */

    /*
    println!("Clone");
    let handleK = handleJ.clone();
    let handleL = handleK * 0.1f32;

    let handleM = handleL.apply_function(ElementFunction::Sigmoid);

    // let C = &A * &B;
    */

    let R = MetalMatrixHandle::to_matrix(&handleA);
    println!("Result of multiplication:");
    println!("{}", R);

    /*
    let handleC = &handleA + &handleB;

    let C = MetalMatrixHandle::to_matrix(&handleC);
    println!("Result of addition:");
    println!("{}", C); */

    // let B = Matrix::new(5, 5, &|row, col| row as f32 / col as f32);

    // unsafe { pool.autorelease() };

}
