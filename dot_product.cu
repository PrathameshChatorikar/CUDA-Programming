#include <iostream>
#include <cuda_runtime.h>

#define N 1024
#define THREADS_PER_BLOCK 256

__global__ void dotProductKernel(int *A, int *B, int *partialSum) {
    __shared__ int cache[THREADS_PER_BLOCK];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int localIdx = threadIdx.x;

    int temp = 0;
    if (tid < N) {
        temp = A[tid] * B[tid];
    }

    cache[localIdx] = temp;

    // Sync all threads in block
    __syncthreads();

    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (localIdx < stride) {
            cache[localIdx] += cache[localIdx + stride];
        }
        __syncthreads();
    }

    // Write result of each block to global memory
    if (localIdx == 0) {
        partialSum[blockIdx.x] = cache[0];
    }
}

int main() {
    int size = N * sizeof(int);
    int *h_A = new int[N];
    int *h_B = new int[N];
    int result = 0;

    // Fill arrays
    for (int i = 0; i < N; i++) {
        h_A[i] = 1;  // You can change to random or i
        h_B[i] = 2;
    }

    // Allocate device memory
    int *d_A, *d_B, *d_partialSum;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);

    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaMalloc((void**)&d_partialSum, blocks * sizeof(int));

    // Copy to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    dotProductKernel<<<blocks, THREADS_PER_BLOCK>>>(d_A, d_B, d_partialSum);

    // Copy partial sums back to host
    int *h_partialSum = new int[blocks];
    cudaMemcpy(h_partialSum, d_partialSum, blocks * sizeof(int), cudaMemcpyDeviceToHost);

    // Final reduction on CPU
    for (int i = 0; i < blocks; i++) {
        result += h_partialSum[i];
    }

    std::cout << "Dot product = " << result << std::endl;

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_partialSum);
    delete[] h_A;
    delete[] h_B;
    delete[] h_partialSum;

    return 0;
}
