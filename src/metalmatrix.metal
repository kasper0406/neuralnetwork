#include <metal_stdlib>

struct MatrixDescriptor {
    uint rows;
    uint cols;
    device float* data;
};

kernel void add(device const float* A,
                device const float* B,
                device float* R,
                uint index [[thread_position_in_grid]])
{
    R[index] = A[index] + B[index];
}

kernel void sub(device const float* A,
                device const float* B,
                device float* R,
                uint index [[thread_position_in_grid]])
{
    R[index] = A[index] - B[index];
}

kernel void entrywise_product(device const float* A,
                              device const float* B,
                              device float* R,
                              uint index [[thread_position_in_grid]])
{
    R[index] = A[index] * B[index];
}

kernel void scalar_multiply(device float* scalar,
                            device const float* A,
                            device float* R,
                            uint index [[thread_position_in_grid]])
{
    R[index] = *scalar * A[index];
}

kernel void transpose(device const MatrixDescriptor& A [[buffer(0)]],
                      device float* R,
                      uint2 index [[thread_position_in_grid]])
{
    R[index.x * A.rows + index.y] = A.data[index.y * A.cols + index.x];
}

kernel void mul(device const MatrixDescriptor& A [[buffer(0)]],
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
