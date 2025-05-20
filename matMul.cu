#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>

__global__ void grouped_matmul_kernel(
    const float* __restrict__ A,  // [batch, M, K]
    const float* __restrict__ B,  // [batch, K, N]
    float* __restrict__ C,        // [batch, M, N]
    int M, int N, int K, int batch)
{
    int batch_id = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a = A[batch_id * M * K + row * K + k];
            float b = B[batch_id * K * N + k * N + col];
            sum += a * b;
        }
        C[batch_id * M * N + row * N + col] = sum;
    }
}

void launch_grouped_matmul(const float* d_A, const float* d_B, float* d_C,
                           int M, int N, int K, int batch)
{
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + 15) / 16, (M + 15) / 16, batch);
    hipLaunchKernelGGL(grouped_matmul_kernel, numBlocks, threadsPerBlock, 0, 0,
                       d_A, d_B, d_C, M, N, K, batch);
}

int main() {
    const int M = 64, N = 64, K = 64, B = 4;
    size_t sizeA = B * M * K * sizeof(float);
    size_t sizeB = B * K * N * sizeof(float);
    size_t sizeC = B * M * N * sizeof(float);

    std::vector<float> h_A(B * M * K), h_B(B * K * N), h_C(B * M * N);
    for (int i = 0; i < h_A.size(); ++i) h_A[i] = static_cast<float>(rand() % 10);
    for (int i = 0; i < h_B.size(); ++i) h_B[i] = static_cast<float>(rand() % 10);

    float *d_A, *d_B, *d_C;
    hipMalloc(&d_A, sizeA);
    hipMalloc(&d_B, sizeB);
    hipMalloc(&d_C, sizeC);

    hipMemcpy(d_A, h_A.data(), sizeA, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B.data(), sizeB, hipMemcpyHostToDevice);

    launch_grouped_matmul(d_A, d_B, d_C, M, N, K, B);

    hipMemcpy(h_C.data(), d_C, sizeC, hipMemcpyDeviceToHost);

    std::cout << "Sample output of batch 0:" << std::endl;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j)
            std::cout << h_C[i * N + j] << " ";
        std::cout << std::endl;
    }

    hipFree(d_A); hipFree(d_B); hipFree(d_C);
    return 0;
}
