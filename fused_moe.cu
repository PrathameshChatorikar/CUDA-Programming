#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(err) \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(-1); \
    }

__global__ void fused_moe_kernel(const half* __restrict__ input,
                                 const float* __restrict__ gate_scores,
                                 const int* __restrict__ expert_ids,
                                 const half* __restrict__ expert_weights,
                                 half* output,
                                 int batch_size,
                                 int hidden_dim,
                                 int num_experts) {
    int token_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_id >= batch_size) return;

    const half* token_input = input + token_id * hidden_dim;
    int expert_id = expert_ids[token_id];
    float gate = gate_scores[token_id];

    const half* weights = expert_weights + expert_id * hidden_dim * hidden_dim;
    half* token_output = output + token_id * hidden_dim;

    for (int i = 0; i < hidden_dim; ++i) {
        float acc = 0.0f;
        for (int j = 0; j < hidden_dim; ++j) {
            float w = __half2float(weights[i * hidden_dim + j]);
            float x = __half2float(token_input[j]);
            acc += w * x;
        }
        acc = fmaxf(acc, 0.0f); // ReLU
        acc *= gate;
        token_output[i] = __float2half(acc);
    }
}

void run_moe_fused(int batch_size, int hidden_dim, int num_experts) {
    size_t token_bytes = batch_size * hidden_dim * sizeof(half);
    size_t gate_bytes = batch_size * sizeof(float);
    size_t id_bytes = batch_size * sizeof(int);
    size_t weight_bytes = num_experts * hidden_dim * hidden_dim * sizeof(half);

    half* h_input = (half*)malloc(token_bytes);
    float* h_gate = (float*)malloc(gate_bytes);
    int* h_ids = (int*)malloc(id_bytes);
    half* h_weights = (half*)malloc(weight_bytes);
    half* h_output = (half*)malloc(token_bytes);

    for (int i = 0; i < batch_size * hidden_dim; ++i)
        h_input[i] = __float2half(((float) rand() / RAND_MAX) - 0.5f);
    for (int i = 0; i < batch_size; ++i) {
        h_gate[i] = 1.0f;
        h_ids[i] = rand() % num_experts;
    }
    for (int i = 0; i < num_experts * hidden_dim * hidden_dim; ++i)
        h_weights[i] = __float2half(((float) rand() / RAND_MAX) - 0.5f);

    half *d_input, *d_output, *d_weights;
    float* d_gate;
    int* d_ids;

    CHECK_CUDA(cudaMalloc(&d_input, token_bytes));
    CHECK_CUDA(cudaMalloc(&d_output, token_bytes));
    CHECK_CUDA(cudaMalloc(&d_weights, weight_bytes));
    CHECK_CUDA(cudaMalloc(&d_gate, gate_bytes));
    CHECK_CUDA(cudaMalloc(&d_ids, id_bytes));

    CHECK_CUDA(cudaMemcpy(d_input, h_input, token_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_gate, h_gate, gate_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ids, h_ids, id_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weights, h_weights, weight_bytes, cudaMemcpyH
