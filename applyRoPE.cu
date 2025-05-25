#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_SIZE 16
#define NUM_HEADS 8  // Number of attention heads
#define SEQ_LENGTH 128 // Sequence length for input tokens
#define EMBEDDING_DIM 512  // Embedding dimension for queries, keys, values

// CUDA kernel for matrix multiplication (QKV)
__global__ void matMul(float *A, float *B, float *C, int m, int n, int k) {
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (row < m && col < k) {
        float value = 0.0f;
        for (int i = 0; i < n; i++) {
            value += A[row * n + i] * B[i * k + col];
        }
        C[row * k + col] = value;
    }
}

// CUDA kernel to apply Rotary Positional Encoding (RoPE) to queries and keys
__global__ void applyRoPE(float *Q, float *K, int seq_len, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int head_dim = dim / NUM_HEADS;

    if (idx < seq_len * head_dim) {
        int row = idx / head_dim;
        int col = idx % head_dim;

        // Calculate the positional encoding angle (rotary frequency)
        float angle = static_cast<float>(row) * M_PI / static_cast<float>(seq_len);
        float cos_val = cos(angle);
        float sin_val = sin(angle);

        // Apply RoPE to the Q and K
        float Q_val = Q[idx];
        float K_val = K[idx];

        Q[idx] = Q_val * cos_val - K_val * sin_val;
        K[idx] = K_val * cos_val + Q_val * sin_val;
    }
}

// CUDA kernel for softmax (for attention scores)
__global__ void softmaxKernel(float *input, float *output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        // Compute max value in the row
        float max_val = -1e20;
        for (int i = 0; i < cols; i++) {
            max_val = fmaxf(max_val, input[row * cols + i]);
        }

        // Compute exponentials and sum
        float sum = 0.0f;
        for (int i = 0; i < cols; i++) {
            sum += expf(input[row * cols + i] - max_val);
        }

        // Store the softmax result
        output[row * cols + col] = expf(input[row * cols + col] - max_val) / sum;
    }
}

// CUDA kernel for the final attention operation
__global__ void attention(float *Q, float *K, float *V, float *output, int seq_len, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < seq_len) {
        float result = 0.0f;
        for (int i = 0; i < seq_len; i++) {
            result += Q[idx * dim + i] * K[i * dim + idx]; // Scaled dot product attention
        }
        output[idx] = result;
    }
}

int main() {
    // Allocate memory for queries, keys, values (QKV) on host and device
    int m = SEQ_LENGTH;  // Number of tokens in the sequence
    int n = EMBEDDING_DIM;  // Embedding dimension
    int k = SEQ_LENGTH;  // Sequence length for the final attention result

    // Host memory allocation
    float *h_Q = (float *)malloc(m * n * sizeof(float));
    float *h_K = (float *)malloc(m * n * sizeof(float));
    float *h_V = (float *)malloc(m * n * sizeof(float));
    float *h_output = (float *)malloc(m * n * sizeof(float));

    // Initialize Q, K, V with random values (simulating embeddings)
    for (int i = 0; i < m * n; ++i) {
        h_Q[i] = static_cast<float>(rand()) / RAND_MAX;
        h_K[i] = static_cast<float>(rand()) / RAND_MAX;
        h_V[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Device memory allocation
    float *d_Q, *d_K, *d_V, *d_output;
    cudaMalloc(&d_Q, m * n * sizeof(float));
    cudaMalloc(&d_K, m * n * sizeof(float));
    cudaMalloc(&d_V, m * n * sizeof(float));
    cudaMalloc(&d_output, m * n * sizeof(float));

    cudaMemcpy(d_Q, h_Q, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, m * n * sizeof(float), cudaMemcpyHostToDevice);

    // Apply RoPE to Q and K
    int threadsPerBlock = 256;
    int blocksPerGrid = (SEQ_LENGTH * EMBEDDING_DIM + threadsPerBlock - 1) / threadsPerBlock;
    applyRoPE<<<blocksPerGrid, threadsPerBlock>>>(d_Q, d_K, SEQ_LENGTH, EMBEDDING_DIM);
    cudaDeviceSynchronize();

    // Perform attention operation (dot product between Q and K, then multiply by V)
    attention<<<blocksPerGrid, threadsPerBlock>>>(d_Q, d_K, d_V, d_output, SEQ_LENGTH, EMBEDDING_DIM);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_output, d_output, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Print a small portion of the result
    for (int i = 0; i < 10; ++i) {
        std::cout << "Output[" << i << "] = " << h_output[i] << std::endl;
    }

    // Clean up
    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_output);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);

    return 0;
}
