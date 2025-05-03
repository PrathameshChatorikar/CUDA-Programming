#include <iostream>
#include <cuda_runtime.h>

#define M 4
#define N 4
#define P 4

__global__ void matMulKernel(int *A, int *B, int *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < P) {
        int sum = 0;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * P + col];
        }
        C[row * P + col] = sum;
    }
}

int main() {
    int sizeA = M * N * sizeof(int);
    int sizeB = N * P * sizeof(int);
    int sizeC = M * P * sizeof(int);

    int h_A[M * N], h_B[N * P], h_C[M * P];

    // Fill A and B with sample values
    for (int i = 0; i < M * N; i++) h_A[i] = 1;
    for (int i = 0; i < N * P; i++) h_B[i] = 2;

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((P + 15) / 16, (M + 15) / 16);
    matMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);

    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // Print result
    std::cout << "Matrix C (Result):" << std::endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < P; j++) {
            std::cout << h_C[i * P + j] << " ";
        }
        std::cout << std::endl;
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
