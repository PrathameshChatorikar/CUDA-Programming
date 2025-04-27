#include <iostream>
#include <cuda_runtime.h>

// CUDA kernel to increment each array element
__global__ void incrementArray(int *arr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        arr[idx] += 1;
    }
}

int main() {
    int N = 1000000; // Make array bigger for better timing

    int *h_array = new int[N];
    for (int i = 0; i < N; ++i) {
        h_array[i] = i;
    }

    int *d_array;
    cudaMalloc(&d_array, N * sizeof(int));
    cudaMemcpy(d_array, h_array, N * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start);

    // Launch the kernel
    incrementArray<<<blocksPerGrid, threadsPerBlock>>>(d_array, N);

    // Record the stop event
    cudaEventRecord(stop);

    // Wait until stop event completes
    cudaEventSynchronize(stop);

    // Calculate elapsed time between events
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy the result back to host
    cudaMemcpy(h_array, d_array, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print benchmark result
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    // (Optional) Print few elements to confirm correctness
    std::cout << "First 5 elements after incrementing: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << h_array[i] << " ";
    }
    std::cout << std::endl;

    // Free memory
    cudaFree(d_array);
    delete[] h_array;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
