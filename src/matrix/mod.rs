pub mod matrix;
pub mod matrixhandle;
pub mod simplematrixhandle;
#[cfg(feature = "metalmatrix")] pub mod metalmatrixhandle;
#[cfg(feature = "cudamatrix")] pub mod cudamatrixhandle;
pub mod verifyingmatrixhandle;

#[cfg(feature = "metalmatrix")] pub type MatrixHandleType = metalmatrixhandle::MetalMatrixHandle;
#[cfg(feature = "cudamatrix")] pub type MatrixHandleType = cudamatrixhandle::CudaMatrixHandle;
#[cfg(not(any(feature = "metalmatrix", feature = "cudamatrix")))] pub type MatrixHandleType = simplematrixhandle::SimpleMatrixHandle;
