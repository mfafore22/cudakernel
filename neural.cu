#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 10
#define BATCH_SIZE 8
#define EPOCHS 10
#define LEARNING_RATE 0.01f
#define TRAIN_SIZE 10000
#define TEST_SIZE 10000

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

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, \
                    (int)status); \
            cudaDeviceReset(); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// All d_* are device pointers. Matrices are row-major.
typedef struct {
    cublasHandle_t cublas_handle;
    float *d_weights1, *d_weights2, *d_bias1, *d_bias2;   // [IN x HID], [HID x OUT], [HID], [OUT]
    float *d_grad_weights1, *d_grad_weights2;             // same shapes as weights
    float *d_grad_bias1, *d_grad_bias2;                   // same shapes as biases
    float *d_input_batch;                                 // [BATCH x INPUT_SIZE]
    float *d_fc1_output;                                  // [BATCH x HIDDEN_SIZE] (after ReLU)
    float *d_fc2_output;                                  // [BATCH x OUTPUT_SIZE] (logits)
} NeuralNetworkCUDA;

// Host-side init helpers (same as neural.c) — fill on CPU, then copy to GPU
void initialize_weights(float *weights, int input_size, int output_size) {
    float scale = sqrtf(6.0f / input_size);
    for (int i = 0; i < input_size * output_size; i++) {
        weights[i] = ((float)rand() / RAND_MAX) * 2.0f * scale - scale;
    }
}

void initialize_bias(float *bias, int size) {
    for (int i = 0; i < size; i++) {
        bias[i] = 0.0f;
    }
}

// Declared here because initialize_neural_network calls it before its definition
void initialize_random_weights(NeuralNetworkCUDA *nn);

void initialize_neural_network(NeuralNetworkCUDA *nn) {
    CUBLAS_CHECK(cublasCreate(&nn->cublas_handle));

    CUDA_CHECK(cudaMalloc(&nn->d_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_bias1, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_bias2, OUTPUT_SIZE * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&nn->d_grad_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_bias1, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_bias2, OUTPUT_SIZE * sizeof(float)));

    // Activation buffers sized for the largest batch we will run
    CUDA_CHECK(cudaMalloc(&nn->d_input_batch, BATCH_SIZE * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_fc1_output, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_fc2_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));

    initialize_random_weights(nn);
}

void initialize_random_weights(NeuralNetworkCUDA *nn) {
    float *h_weights1 = (float *)malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    initialize_weights(h_weights1, INPUT_SIZE, HIDDEN_SIZE);
    CUDA_CHECK(cudaMemcpy(nn->d_weights1, h_weights1,
        INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    free(h_weights1);

    float *h_weights2 = (float *)malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    initialize_weights(h_weights2, HIDDEN_SIZE, OUTPUT_SIZE);
    CUDA_CHECK(cudaMemcpy(nn->d_weights2, h_weights2,
        HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    free(h_weights2);

    float *h_bias1 = (float *)malloc(HIDDEN_SIZE * sizeof(float));
    initialize_bias(h_bias1, HIDDEN_SIZE);
    CUDA_CHECK(cudaMemcpy(nn->d_bias1, h_bias1,
        HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    free(h_bias1);

    float *h_bias2 = (float *)malloc(OUTPUT_SIZE * sizeof(float));
    initialize_bias(h_bias2, OUTPUT_SIZE);
    CUDA_CHECK(cudaMemcpy(nn->d_bias2, h_bias2,
        OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    free(h_bias2);
}

void free_neural_network(NeuralNetworkCUDA *nn) {
    CUDA_CHECK(cudaFree(nn->d_weights1));
    CUDA_CHECK(cudaFree(nn->d_weights2));
    CUDA_CHECK(cudaFree(nn->d_bias1));
    CUDA_CHECK(cudaFree(nn->d_bias2));
    CUDA_CHECK(cudaFree(nn->d_grad_weights1));
    CUDA_CHECK(cudaFree(nn->d_grad_weights2));
    CUDA_CHECK(cudaFree(nn->d_grad_bias1));
    CUDA_CHECK(cudaFree(nn->d_grad_bias2));
    CUDA_CHECK(cudaFree(nn->d_input_batch));
    CUDA_CHECK(cudaFree(nn->d_fc1_output));
    CUDA_CHECK(cudaFree(nn->d_fc2_output));
    CUBLAS_CHECK(cublasDestroy(nn->cublas_handle));
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
    // One block per sample, and only thread 0 does the work.
    // Without the threadIdx guard, a launch with >1 thread would race on x.
    int b = blockIdx.x;
    if (b < batch_size && threadIdx.x == 0) {
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

void forward_pass_only(NeuralNetworkCUDA *nn, int batch_size) {
        const float alpha = 1.0f, beta = 0.0f;

        CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                               HIDDEN_SIZE, batch_size, INPUT_SIZE,
                               &alpha, nn->d_weights1, HIDDEN_SIZE,
                               nn->d_input_batch, INPUT_SIZE, &beta,
                               nn->d_fc1_output,
                               HIDDEN_SIZE));

        int total_hidden = batch_size * HIDDEN_SIZE;
        int grid_hidden = (total_hidden + 255) / 256;
        bias_forward_kernel<<<grid_hidden, 256>>>(nn->d_fc1_output, nn->d_bias1,
        batch_size, HIDDEN_SIZE);

        relu_forward_kernel<<<grid_hidden, 256>>>(
            nn->d_fc1_output, total_hidden);

        CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                               OUTPUT_SIZE, batch_size, HIDDEN_SIZE,
                               &alpha, nn->d_weights2, OUTPUT_SIZE,
                               nn->d_fc1_output, HIDDEN_SIZE, &beta,
                               nn->d_fc2_output,
                               OUTPUT_SIZE));

        int total_out = batch_size * OUTPUT_SIZE;
        int grid_out = (total_out + 255) / 256;
        bias_forward_kernel<<<grid_out, 256>>>(nn->d_fc2_output, nn->d_bias2,
        batch_size, OUTPUT_SIZE);
        CUDA_CHECK(cudaDeviceSynchronize());
}
