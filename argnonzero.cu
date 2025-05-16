#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void count_nonzero(const int* input, int* count, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && input[idx] != 0) {
        atomicAdd(count, 1);
    }
}

__global__ void collect_nonzero_indices(const int* input, int* output, int* pos, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && input[idx] != 0) {
        int out_idx = atomicAdd(pos, 1);
        output[out_idx] = idx;
    }
}

void cuda_argnonzero(const std::vector<int>& host_input, std::vector<int>& host_output) {
    int size = host_input.size();
    int *d_input, *d_output, *d_count;

    cudaMalloc(&d_input, size * sizeof(int));
    cudaMemcpy(d_input, host_input.data(), size * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_count, sizeof(int));
    cudaMemset(d_count, 0, sizeof(int));

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    // Count non-zero elements
    count_nonzero<<<blocks, threads>>>(d_input, d_count, size);
    cudaDeviceSynchronize();

    int count;
    cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    cudaMalloc(&d_output, count * sizeof(int));
    cudaMemset(d_count, 0, sizeof(int));  // reuse d_count as position counter

    // Collect non-zero indices
    collect_nonzero_indices<<<blocks, threads>>>(d_input, d_output, d_count, size);
    cudaDeviceSynchronize();

    host_output.resize(count);
    cudaMemcpy(host_output.data(), d_output, count * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_count);
}

int main() {
    std::vector<int> input = {0, 4, 0, 5, 6, 0, 7};
    std::vector<int> output;

    cuda_argnonzero(input, output);

    std::cout << "Indices of non-zero elements: ";
    for (int idx : output) std::cout << idx << " ";
    std::cout << std::endl;

    return 0;
}
