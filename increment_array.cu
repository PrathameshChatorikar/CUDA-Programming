// increment_array.cu
// CUDA program to increment each element of an array by 1

#include <iostream>
using namespace std;

// CUDA kernel function to increment each element
__global__ void increment(int *arr, int n) {
    // Each thread calculates its own index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure we don't access out of bounds
    if (idx < n) {
        arr[idx] = arr[idx] + 1;
    }
}

int main() {
    // Number of elements in the array
    int n = 10;

    // Allocate memory for array on CPU (host)
    int *h_arr = new int[n];

    // Initialize array with some values
    for (int i = 0; i < n; ++i) {
        h_arr[i] = i;
    }

    // Allocate memory for array on GPU (device)
    int *d_arr;
    cudaMalloc(&d_arr, n * sizeof(int));

    // Copy data from CPU to GPU
    cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice);

    // Define block size and grid size
    int blockSize = 256;                      // Number of threads in a block
    int gridSize = (n + blockSize - 1) / blockSize; // Number of blocks needed

    // Launch the kernel
    increment<<<gridSize, blockSize>>>(d_arr, n);

    // Copy the result back from GPU to CPU
    cudaMemcpy(h_arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the updated array
    cout << "Incremented array:" << endl;
    for (int i = 0; i < n; ++i) {
        cout << h_arr[i] << " ";
    }
    cout << endl;

    // Free the memory on GPU and CPU
    cudaFree(d_arr);
    delete[] h_arr;

    return 0;
}
