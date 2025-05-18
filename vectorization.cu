__global__ void matrix_mult_optimized(float *A, float *B, float *C, int N) {
    extern __shared__ float shared_mem[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int aBegin = N * 32 * by;
    int aEnd = aBegin + N - 1;
    int bBegin = 32 * bx;
    int bEnd = bBegin + N - 1;

    float Cvalue = 0.0f;
    float *shared_A = shared_mem;
    float *shared_B = &shared_mem[32 * 32];

    for (int a = aBegin, b = bBegin; a <= aEnd; a += 32, b += 32) {
        // Load data into shared memory
        shared_A[ty * 32 + tx] = A[a + N * ty + tx];
        shared_B[ty * 32 + tx] = B[b + N * ty + tx];

        __syncthreads();

        // Perform the computation using the loaded data
        for (int k = 0; k < 32; ++k) {
            Cvalue += shared_A[ty * 32 + k] * shared_B[k * 32 + tx];
        }
        __syncthreads();
    }

    // Write the result back to global memory
    int cIdx = N * 32 * by + 32 * bx + N * ty + tx;
    C[cIdx] = Cvalue;
}
