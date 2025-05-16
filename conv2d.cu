__global__ void conv2d_kernel(const float* input, const float* kernel, float* output,
                              int width, int height, int ksize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // column
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // row

    int out_width = width - ksize + 1;
    int out_height = height - ksize + 1;

    if (x < out_width && y < out_height) {
        float sum = 0.0f;
        for (int ky = 0; ky < ksize; ++ky) {
            for (int kx = 0; kx < ksize; ++kx) {
                int ix = x + kx;
                int iy = y + ky;
                float i_val = input[iy * width + ix];
                float k_val = kernel[(ksize - 1 - ky) * ksize + (ksize - 1 - kx)];  // flip kernel
                sum += i_val * k_val;
            }
        }
        output[y * out_width + x] = sum;
    }
}
#include <iostream>
#include <vector>

void run_conv2d() {
    const int width = 5, height = 5;
    const int ksize = 3;
    const int out_width = width - ksize + 1;
    const int out_height = height - ksize + 1;

    float h_input[width * height] = {
         1,  2,  3,  4,  5,
         6,  7,  8,  9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25
    };

    float h_kernel[ksize * ksize] = {
         1, 0, -1,
         1, 0, -1,
         1, 0, -1
    };  // Sobel edge detection (horizontal)

    float h_output[out_width * out_height];

    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, sizeof(float) * width * height);
    cudaMalloc(&d_kernel, sizeof(float) * ksize * ksize);
    cudaMalloc(&d_output, sizeof(float) * out_width * out_height);

    cudaMemcpy(d_input, h_input, sizeof(float) * width * height, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, sizeof(float) * ksize * ksize, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((out_width + threads.x - 1) / threads.x, (out_height + threads.y - 1) / threads.y);
    conv2d_kernel<<<blocks, threads>>>(d_input, d_kernel, d_output, width, height, ksize);

    cudaMemcpy(h_output, d_output, sizeof(float) * out_width * out_height, cudaMemcpyDeviceToHost);

    std::cout << "Output:\n";
    for (int y = 0; y < out_height; ++y) {
        for (int x = 0; x < out_width; ++x) {
            std::cout << h_output[y * out_width + x] << "\t";
        }
        std::cout << "\n";
    }

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}
