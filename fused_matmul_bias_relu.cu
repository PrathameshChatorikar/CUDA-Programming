#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

#define IDX2C(i, j, ld) (((j)*(ld))+(i))  // Column-major index (can adjust for row-major)

// Fused kernel: C = ReLU(A * B + bias)
__global__ void fused_matmul_bias_relu(float* A, float* B, float* bias, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // M dimension
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // N dimension

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        sum += bias[col];           // Add bias
        C[row * N + col] = fmaxf(sum, 0.0f);  // ReLU
    }
}

int main() {
    // Example sizes
    const int M = 2, N = 3, K = 4;

    // Host allocations
    float h_A[M*K] = { /* fill with your data */ };
    float h_B[K*N] = { /* fill with your data */ };
    float h_bias[N] = { /* fill with your data */ };
    float h_C[M*N];

    // Device allocations
    float *d_A, *d_B, *d_bias, *d_C;
    cudaMalloc(&d_A, M*K * sizeof(float));
    cudaMalloc(&d_B, K*N * sizeof(float));
    cudaMalloc(&d_bias, N * sizeof(float));
    cudaMalloc(&d_C, M*N * sizeof(float));

    // Copy inputs to device
    cudaMemcpy(d_A, h_A, M*K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K*N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, N * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel launch config
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    // Launch fused kernel
    fused_matmul_bias_relu<<<gridDim, blockDim>>>(d_A, d_B, d_bias, d_C, M, N, K);

    // Copy result back
    cudaMemcpy(h_C, d_C, M*N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print result
    std::cout << "Output C:\n";
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << h_C[i * N + j] << " ";
        }
        std::cout << "\n";
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_bias);
    cudaFree(d_C);

    return 0;
}
