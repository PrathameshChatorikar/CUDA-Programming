#include <cuda_runtime.h>
#include <stdio.h>

#define N 1024  // size of input

__global__ void vector_sum_kernel(float *input, float *result, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Load input into shared memory
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Write result from block 0
    if (tid == 0)
        atomicAdd(result, sdata[0]);
}

int main() {
    float *h_data = new float[N];
    float h_result = 0.0f;

    for (int i = 0; i < N; ++i)
        h_data[i] = 1.0f;  // Example: all elements = 1.0 → sum = N

    float *d_data, *d_result;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(float));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    vector_sum_kernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_data, d_result, N);

    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Sum = %f\n", h_result);

    cudaFree(d_data);
    cudaFree(d_result);
    delete[] h_data;

    return 0;
}
