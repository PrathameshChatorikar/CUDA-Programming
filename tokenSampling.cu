#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>

__global__ void init_rng(curandState* states, unsigned long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void top_mass_peak_sampling(const float* probs, int vocab_size, int* sampled_idx, curandState* states) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid == 0) {
        float total_score = 0.0f;
        float scores[8192];  // adjust size to max vocab

        for (int i = 0; i < vocab_size; ++i) {
            float left = (i > 0) ? probs[i - 1] : 0.0f;
            float right = (i < vocab_size - 1) ? probs[i + 1] : 0.0f;

            // Peak boost: higher if local max
            float peak_boost = (probs[i] > left && probs[i] > right) ? 1.3f : 1.0f;

            scores[i] = probs[i] * peak_boost;
            total_score += scores[i];
        }

        // Normalize scores
        for (int i = 0; i < vocab_size; ++i)
            scores[i] /= total_score;

        // Sample
        float r = curand_uniform(&states[0]);
        float cum_sum = 0.0f;
        for (int i = 0; i < vocab_size; ++i) {
            cum_sum += scores[i];
            if (r <= cum_sum) {
                *sampled_idx = i;
                break;
            }
        }
    }
}
