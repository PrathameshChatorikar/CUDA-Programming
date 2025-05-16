#include <cuda_runtime.h>
#include <stdio.h>

__global__ void conv1d_kernel(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int output_size = input_size - kernel_size + 1;

    if (tid < output_size) {
        float sum = 0.0f;
        for (int k = 0; k < kernel_size; ++k) {
            sum += input[tid + k] * kernel[kernel_size - 1 - k];  // Flip kernel for convolution
        }
        output[tid] = sum;
    }
}
#include <iostream>
#include <vector>

void run_conv1d() {
    const int input_size = 10;
    const int kernel_size = 3;
    const int output_size = input_size - kernel_size + 1;

    float h_input[input_size]  = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    float h_kernel[kernel_size] = {1, 0, -1};  // Simple gradient kernel
    float h_output[output_size];

    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_kernel, kernel_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));

    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (output_size + blockSize - 1) / blockSize;
    conv1d_kernel<<<gridSize, blockSize>>>(d_input, d_kernel, d_output, input_size, kernel_size);

    cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Output: ";
    for (int i = 0; i < output_size; ++i)
        std::cout << h_output[i] << " ";
    std::cout << std::endl;

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}
