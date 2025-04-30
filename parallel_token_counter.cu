// parallel_token_counter.cu
#include <iostream>
#include <cuda.h>

__global__ void countToken(int* input, int* count, int token, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (input[idx] == token) {
        atomicAdd(count, 1);  // safely increment counter
    }
}

int main() {
    const int N = 1024;
    int token = 42; // "target token" to count
    int host_input[N];
    int host_count = 0;

    // Fill array with random values
    for (int i = 0; i < N; i++) {
        host_input[i] = rand() % 100;
    }

    int* dev_input;
    int* dev_count;

    cudaMalloc((void**)&dev_input, N * sizeof(int));
    cudaMalloc((void**)&dev_count, sizeof(int));
    cudaMemcpy(dev_input, host_input, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_count, &host_count, sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    countToken<<<blocks, threadsPerBlock>>>(dev_input, dev_count, token, N);
    cudaMemcpy(&host_count, dev_count, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Token " << token << " found " << host_count << " times in the array." << std::endl;

    cudaFree(dev_input);
    cudaFree(dev_count);
    return 0;
}
