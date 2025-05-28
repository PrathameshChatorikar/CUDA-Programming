#include <iostream>
#include <cuda_runtime.h>

// CUDA kernel for vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;  // Calculate thread index
    if (index < N) {
        C[index] = A[index] + B[index];  // Perform element-wise addition
    }
}

int main() {
    int N = 1 << 20;  // Number of elements in the vector (2^20 = 1048576 elements)
    size_t size = N * sizeof(float);  // Size of memory to allocate (in bytes)

    // Allocate memory for vectors on the host (CPU)
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize vectors A and B with sample values
    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Allocate memory for vectors on the device (GPU)
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define number of threads and blocks
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel for vector addition
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Check for any errors in kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy the result vector back to the host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify the result
    for (int i = 0; i < N; i++) {
        if (h_C[i] != 3.0f) {
            std::cerr << "Error: Element " << i << " is " << h_C[i] << " instead of 3.0" << std::endl;
            return -1;
        }
    }

    std::cout << "Vector addition successful!" << std::endl;

    // Free the allocated memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
