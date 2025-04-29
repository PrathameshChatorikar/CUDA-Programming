#include <iostream>
#include <cuda_runtime.h>

// CUDA kernel to perform sum reduction
__global__ void reduce_sum(int *input, int *output, int n) {
    extern __shared__ int sdata[];

    int tid = threadIdx.x;                          // Local thread ID within the block
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // Global index in the input array

    // Step 1: Load input into shared memory (or 0 if out of bounds)
    sdata[tid] = (i < n) ? input[i] : 0;
    __syncthreads();  // Ensure all threads have loaded their data

    // Step 2: Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];  // Each thread adds its neighbor
        }
        __syncthreads();  // Synchronize before next reduction step
    }

    // Step 3: Write the block's result to output array
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

int main() {
    const int N = 8;
    int h_input[N] = {13, 27, 15, 14, 33, 2, 24, 6};  // Input array on host
    int h_output;                                    // Variable to store final sum

    int *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(int));            // Allocate memory on device for input
    cudaMalloc(&d_output, sizeof(int));               // Allocate memory on device for output

    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);  // Copy input data to device

    // Launch kernel: 1 block, 8 threads, and shared memory for 8 integers
    reduce_sum<<<1, N, N * sizeof(int)>>>(d_input, d_output, N);

    cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);  // Copy result back to host

    std::cout << "Total sum = " << h_output << std::endl;  // Print the final result (should be 134)

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
