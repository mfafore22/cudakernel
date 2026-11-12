#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            cudaDeviceReset(); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

void initialize_neural_network(NeuralNetwork *nn) {
    CUDA_CHECK(cudaMalloc(&nn->weights1, INPUT_SIZE * HIDDEN_SIZE *
    sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->weights2, HIDDEN_SIZE * OUTPUT_SIZE *
    sizeof(float)));
        CUDA_CHECK(cudaMalloc(&nn->bias1, HIDDEN_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&nn->bias2, OUTPUT_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&nn->grad_weights1, INPUT_SIZE * HIDDEN_SIZE
        * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&nn->grad_weights2, HIDDEN_SIZE * OUTPUT_SIZE
        * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&nn->grad_bias1, HIDDEN_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&nn->grad_bias2, OUTPUT_SIZE * sizeof(float)));
        initialize_random_weights(nn);
}

void initialize_random_weights(NeuralNetwork *nn) {
    float *h_weights1 = (float *)malloc(INPUT_SIZE * HIDDEN_SIZE
    * sizeof(float));
        initialize_weights(h_weights1, INPUT_SIZE, HIDDEN_SIZE);
        CUDA_CHECK(cudaMemcpy(nn->weights1, h_weights1, INPUT_SIZE
        * HIDDEN_SIZE * sizeof(float),
        cudaMemcpyHostToDevice));
        free(h_weights1);

        float *h_weights2 = (float *)malloc(HIDDEN_SIZE * OUTPUT_SIZE
        * sizeof(float));
        initialize_weights(h_weights2, HIDDEN_SIZE, OUTPUT_SIZE);
        CUDA_CHECK(cudaMemcpy(nn->weights2, h_weights2, HIDDEN_SIZE
        * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        free(h_weights2);

        float *h_bias1 = (float *)malloc(HIDDEN_SIZE * sizeof(float));
        initialize_bias(h_bias1, HIDDEN_SIZE);
        CUDA_CHECK(cudaMemcpy(nn->bias1, h_bias1, HIDDEN_SIZE
        * sizeof(float), cudaMemcpyHostToDevice));
        free(h_bias1);

        float *h_bias2 = (float *)malloc(OUTPUT_SIZE * sizeof(float));
        initialize_bias(h_bias2, OUTPUT_SIZE);
        CUDA_CHECK(cudaMemcpy(nn->bias2, h_bias2, OUTPUT_SIZE
        * sizeof(float), cudaMemcpyHostToDevice));
        free(h_bias2);
}


