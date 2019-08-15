#include <metal_stdlib>

kernel void add_arrays(device const float* A,
                       device const float* B,
                       device float* result,
                       uint index [[thread_position_in_grid]])
{
    result[index] = A[index] + B[index];
}

struct MatrixDescriptor {
    uint rows;
    uint cols;
    device float* data;
};

kernel void multiply(device const MatrixDescriptor& A [[buffer(0)]],
                     device const MatrixDescriptor& B [[buffer(1)]],
                     device float* R,
                     uint2 index [[thread_position_in_grid]])
{
    float scalar = 0;
    for (uint k = 0; k < A.cols; k++) {
        scalar += A.data[index.y * A.cols + k] * B.data[k * B.cols + index.x];
    }
    R[index.y * B.cols + index.x] = scalar;
}
