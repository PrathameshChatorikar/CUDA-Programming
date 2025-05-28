#include <iostream>
#include <cuda_runtime.h>

#define INPUT_SIZE 3  // Number of input neurons
#define HIDDEN_SIZE 4  // Number of hidden neurons
#define OUTPUT_SIZE 1  // Number of output neurons
#define LEARNING_RATE 0.01

// Sigmoid activation function
__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

// Sigmoid derivative function
__device__ float sigmoid_derivative(float x) {
    return x * (1 - x);
}

// Kernel to perform feedforward propagation
__global__ void feedforward(float* inputs, float* weights1, float* weights2, float* hidden, float* output, int input_size, int hidden_size, int output_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < hidden_size) {
        float sum = 0;
        for (int j = 0; j < input_size; j++) {
            sum += inputs[j] * weights1[i * input_size + j];  // Weighted sum for hidden layer
        }
        hidden[i] = sigmoid(sum);  // Apply sigmoid activation
    }

    if (i < output_size) {
        float sum = 0;
        for (int j = 0; j < hidden_size; j++) {
            sum += hidden[j] * weights2[i * hidden_size + j];  // Weighted sum for output layer
        }
        output[i] = sigmoid(sum);  // Apply sigmoid activation
    }
}

// Kernel for backpropagation and weight update
__global__ void backpropagate(float* inputs, float* hidden, float* output, float* weights1, float* weights2, float* d_weights1, float* d_weights2, float* target, int input_size, int hidden_size, int output_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < output_size) {
        float error = target[i] - output[i];  // Calculate error
        float d_output = error * sigmoid_derivative(output[i]);  // Gradient for output layer

        // Update weights2
        for (int j = 0; j < hidden_size; j++) {
            atomicAdd(&d_weights2[i * hidden_size + j], d_output * hidden[j]);  // Gradient descent update for weights2
        }

        // Backpropagate error to hidden layer
        for (int j = 0; j < hidden_size; j++) {
            atomicAdd(&d_weights1[j * input_size + i], d_output * weights2[i * hidden_size + j] * sigmoid_derivative(hidden[j]) * inputs[j]);  // Update weights1
        }
    }
}

// Function to train the network
void train(float* inputs, float* target, float* weights1, float* weights2, int input_size, int hidden_size, int output_size, int epochs) {
    float *d_inputs, *d_target, *d_weights1, *d_weights2, *d_hidden, *d_output, *d_d_weights1, *d_d_weights2;
    
    cudaMalloc((void**)&d_inputs, input_size * sizeof(float));
    cudaMalloc((void**)&d_target, output_size * sizeof(float));
    cudaMalloc((void**)&d_weights1, hidden_size * input_size * sizeof(float));
    cudaMalloc((void**)&d_weights2, output_size * hidden_size * sizeof(float));
    cudaMalloc((void**)&d_hidden, hidden_size * sizeof(float));
    cudaMalloc((void**)&d_output, output_size * sizeof(float));
    cudaMalloc((void**)&d_d_weights1, hidden_size * input_size * sizeof(float));
    cudaMalloc((void**)&d_d_weights2, output_size * hidden_size * sizeof(float));

    cudaMemcpy(d_inputs, inputs, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target, output_size * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize weights (you would typically use random initialization)
    float h_weights1[hidden_size * input_size] = {0.5f, -0.3f, 0.2f, -0.4f, 0.1f, -0.5f, 0.6f, 0.3f, -0.2f, 0.1f, 0.7f, -0.6f};  // Example weights for simplicity
    float h_weights2[output_size * hidden_size] = {0.8f, -0.5f, 0.2f, 0.7f};  // Example weights for simplicity

    cudaMemcpy(d_weights1, h_weights1, hidden_size * input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights2, h_weights2, output_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice);

    // Training loop
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Feedforward
        feedforward<<<1, hidden_size>>>(d_inputs, d_weights1, d_weights2, d_hidden, d_output, input_size, hidden_size, output_size);
        cudaDeviceSynchronize();

        // Backpropagate and update weights
        backpropagate<<<1, output_size>>>(d_inputs, d_hidden, d_output, d_weights1, d_weights2, d_d_weights1, d_d_weights2, d_target, input_size, hidden_size, output_size);
        cudaDeviceSynchronize();

        // Apply the gradient updates to weights
        cudaMemcpy(h_weights1, d_d_weights1, hidden_size * input_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_weights2, d_d_weights2, output_size * hidden_size * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Here you would typically apply the learning rate and update the weights

        std::cout << "Epoch: " << epoch << " Loss: " << "..." << std::endl;
    }

    // Clean up
    cudaFree(d_inputs);
    cudaFree(d_target);
    cudaFree(d_weights1);
    cudaFree(d_weights2);
    cudaFree(d_hidden);
    cudaFree(d_output);
    cudaFree(d_d_weights1);
    cudaFree(d_d_weights2);
}

int main() {
    // Example input (3 features) and target (1 output)
    float inputs[INPUT_SIZE] = {0.5f, 0.2f, 0.8f};
    float target[OUTPUT_SIZE] = {1.0f};

    // Initialize weights and other parameters
    float weights1[HIDDEN_SIZE * INPUT_SIZE];
    float weights2[OUTPUT_SIZE * HIDDEN_SIZE];
    
    // Train the model
    train(inputs, target, weights1, weights2, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, 100);

    return 0;
}
