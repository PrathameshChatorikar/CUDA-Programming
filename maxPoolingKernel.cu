__global__ void max_pooling_kernel(float* input, float* output, int input_width, int input_height, int output_width, int output_height, int pool_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < output_width && y < output_height) {
        int start_x = x * pool_size;
        int start_y = y * pool_size;

        float max_val = -FLT_MAX;
        for (int i = 0; i < pool_size; ++i) {
            for (int j = 0; j < pool_size; ++j) {
                int ix = start_x + i;
                int iy = start_y + j;
                if (ix < input_width && iy < input_height) {
                    int idx = iy * input_width + ix;
                    max_val = fmaxf(max_val, input[idx]);
                }
            }
        }
        output[y * output_width + x] = max_val;
    }
}

void max_pooling(float* input, float* output, int input_width, int input_height, int pool_size) {
    int output_width = input_width / pool_size;
    int output_height = input_height / pool_size;

    float* d_input;
    float* d_output;

    cudaMalloc(&d_input, input_width * input_height * sizeof(float));
    cudaMalloc(&d_output, output_width * output_height * sizeof(float));

    cudaMemcpy(d_input, input, input_width * input_height * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((output_width + blockDim.x - 1) / blockDim.x, (output_height + blockDim.y - 1) / blockDim.y);

    max_pooling_kernel<<<gridDim, blockDim>>>(d_input, d_output, input_width, input_height, output_width, output_height, pool_size);

    cudaMemcpy(output, d_output, output_width * output_height * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
