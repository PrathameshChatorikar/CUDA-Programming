#include <cuda_runtime.h>
#include <iostream>

#define SIZE 1024  // Must be power of two

__global__ void bitonicSortKernel(int* data, int j, int k) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int ixj = i ^ j;

    if (ixj > i) {
        if ((i & k) == 0) {
            if (data[i] > data[ixj]) {
                int temp = data[i];
                data[i] = data[ixj];
                data[ixj] = temp;
            }
        } else {
            if (data[i] < data[ixj]) {
                int temp = data[i];
                data[i] = data[ixj];
                data[ixj] = temp;
            }
        }
    }
}

void bitonicSort(int* d_data, int size) {
    dim3 blocks(size / 1024);
    dim3 threads(1024);

    for (int k = 2; k <= size; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonicSortKernel<<<blocks, threads>>>(d_data, j, k);
            cudaDeviceSynchronize();
        }
    }
}

int main() {
    int h_data[SIZE];
    for (int i = 0; i < SIZE; ++i) h_data[i] = rand() % 1000;

    int* d_data;
    cudaMalloc((void**)&d_data, SIZE * sizeof(int));
    cudaMemcpy(d_data, h_data, SIZE * sizeof(int), cudaMemcpyHostToDevice);

    bitonicSort(d_data, SIZE);

    cudaMemcpy(h_data, d_data, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    std::cout << "Sorted output:\n";
    for (int i = 0; i < 16; ++i) std::cout << h_data[i] << " ";
    std::cout << "...\n";

    return 0;
}
