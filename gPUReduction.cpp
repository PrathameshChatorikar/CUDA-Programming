// Filename: reduce1.cu
// Commit message: Basic GPU reduction using shared memory (Kernel 1)

// This program demonstrates a simple reduction operation on GPU using CUDA.
// It reduces an array of integers by summing them block-wise and storing results in global memory.

#include <stdio.h>
#include <stdlib.h>

// CUDA kernel: performs reduction within each block
__global__ void reduce1(int *g_idata, int *g_odata, unsigned int n) {
    // Shared memory for this block
    extern __shared__ int sdata[];

    // Thread index within the block
    unsigned int tid = threadIdx.x;

    // Global thread index (each thread handles one element)
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input from global memory to shared memory (if within bounds)
    sdata[tid] = (i < n) ? g_idata[i] : 0;

    // Synchronize to make sure all threads loaded their data
    __syncthreads();

    // Perform reduction in shared memory (slow naive version using modulo)
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        // Only threads where tid is multiple of 2s do work
        if ((tid % (2 * s)) == 0 && (tid + s) < blockDim.x) {
            sdata[tid] += sdata[tid + s];
        }
        // Wait for all threads to complete current stage
        __syncthreads();
    }

    // Write the result of the block to global memory (only thread 0)
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

// Host code
int main() {
    const int N = 1024;  // Size of input array
    const int BLOCK_SIZE = 256;

    int *h_idata = (int *)malloc(N * sizeof(int));
    int *h_odata = (int *)malloc((N / BLOCK_SIZE) * sizeof(int));

    // Initialize input data
    for (int i = 0; i < N; ++i) {
        h_idata[i] = 1;  // Set all elements to 1 (easy to check sum = N)
    }

    // Device memory allocation
    int *d_idata, *d_odata;
    cudaMalloc((void **)&d_idata, N * sizeof(int));
    cudaMalloc((void **)&d_odata, (N / BLOCK_SIZE) * sizeof(int));

    // Copy input data to device
    cudaMemcpy(d_idata, h_idata, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    reduce1<<<N / BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_idata, d_odata, N);

    // Copy output data back to host
    cudaMemcpy(h_odata, d_odata, (N / BLOCK_SIZE) * sizeof(int), cudaMemcpyDeviceToHost);

    // Final reduction on CPU
    int total_sum = 0;
    for (int i = 0; i < N / BLOCK_SIZE; ++i) {
        total_sum += h_odata[i];
    }

    // Display result
    printf("Reduced sum = %d\n", total_sum);

    // Free memory
    cudaFree(d_idata);
    cudaFree(d_odata);
    free(h_idata);
    free(h_odata);

    return 0;
}
