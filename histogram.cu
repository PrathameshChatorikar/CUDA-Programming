#include <cuda_runtime.h>
#include <stdio.h>

#define NUM_BINS 16      // number of histogram bins
#define N 1024           // size of input data

__global__ void histogram_kernel(int *data, int *hist, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        int bin = data[idx];
        if (bin >= 0 && bin < NUM_BINS) {
            atomicAdd(&hist[bin], 1);
        }
    }
}

int main() {
    int *h_data = new int[N];
    int *h_hist = new int[NUM_BINS]();

    // Fill input with values between 0 and NUM_BINS-1
    for (int i = 0; i < N; ++i) {
        h_data[i] = rand() % NUM_BINS;
    }

    int *d_data, *d_hist;
    cudaMalloc(&d_data, N * sizeof(int));
    cudaMalloc(&d_hist, NUM_BINS * sizeof(int));
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, NUM_BINS * sizeof(int));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    histogram_kernel<<<gridSize, blockSize>>>(d_data, d_hist, N);

    cudaMemcpy(h_hist, d_hist, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);

    // Print result
    for (int i = 0; i < NUM_BINS; ++i)
        printf("Bin[%d] = %d\n", i, h_hist[i]);

    cudaFree(d_data);
    cudaFree(d_hist);
    delete[] h_data;
    delete[] h_hist;

    return 0;
}
