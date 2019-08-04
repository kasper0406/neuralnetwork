#include <metal_stdlib>

kernel void add_arrays(device const float* A,
                       device const float* B,
                       device float* result,
                       uint index [[thread_position_in_grid]])
{
    result[index] = A[index] + B[index];
}

kernel void multiply(device int A_rows, device int A_cols, device const float* A,
                     device int B_rows, device int B_cols, device const float* B,
                     device float* R,
                     uint2 index [[thread_position_in_grid]])
{
    float scalar = 0;
    for (int k = 0; k < A_cols; k++) {
        scalar += A[index.y * A_cols + k] * B[k * B_cols + index.x];
    }
    R[index.y * B_cols + index.x] = scalar;
}
