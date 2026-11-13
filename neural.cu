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

__global__ void relu_forward_kernel(float *x, int size) { //
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] = fmaxf(0.0f, x[idx]);
    }
}

__global__ void bias_forward_kernel(float *x, float *bias, int batch_size,
int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = idx / size; //
    int i = idx % size;

    if (b < batch_size && i < size) {
        x[idx] += bias[i];
    }
}

__global__ void softmax_kernel(float *x, int batch_size, int size) {
    int b = blockIdx.x; //
    if (b < batch_size) {
        float max_val = x[b * size];
        for (int i = 1; i < size; ++i) {
            max_val = fmaxf(max_val, x[b * size + i]); //
        }

        float sum = 0.0f;
        for (int i = 0; i < size; ++i) {
            x[b * size + i] = expf(x[b * size + i] - max_val);
            sum += x[b * size + i];
        }

        for (int i = 0; i < size; ++i) {
            x[b * size + i] = fmaxf(x[b * size + i] / sum, 1e-7f); //
        }
    }
}


