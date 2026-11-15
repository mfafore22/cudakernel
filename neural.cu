#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 10
#define BATCH_SIZE 8
#define EPOCHS 10
#define LEARNING_RATE 0.01f
#define TRAIN_SIZE 10000
#define TEST_SIZE 10000

typedef struct {
    float *weights1, *weights2, *bias1, *bias2;
    float *grad_weights1, *grad_weights2, *grad_bias1, *grad_bias2;
} NeuralNetwork;

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
void initialize_random_weights(NeuralNetwork *nn);

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

// ---------------------------------------------------------------------------
// Matrix multiply kernels (GPU versions of the CPU matmuls in neural.c)
// All matrices are row-major. Each thread computes ONE output element.
// Launch with 2D grid: x covers columns, y covers rows.
// ---------------------------------------------------------------------------

#define TILE 16

// C[m x k] = A[m x n] @ B[n x k]          (used in forward pass)
__global__ void matmul_a_b_kernel(float *A, float *B, float *C,
int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < k) {
        float sum = 0.0f;   // accumulate in a register, write once
        for (int l = 0; l < n; ++l) {
            sum += A[row * n + l] * B[l * k + col];
        }
        C[row * k + col] = sum;
    }
}

// C[n x k] = A^T @ B,  A is [m x n], B is [m x k]   (weight gradients)
__global__ void matmul_at_b_kernel(float *A, float *B, float *C,
int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // 0..n-1
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // 0..k-1
    if (row < n && col < k) {
        float sum = 0.0f;
        for (int l = 0; l < m; ++l) {
            sum += A[l * n + row] * B[l * k + col];
        }
        C[row * k + col] = sum;
    }
}

// C[m x k] = A @ B^T,  A is [m x n], B is [k x n]   (input gradients)
__global__ void matmul_a_bt_kernel(float *A, float *B, float *C,
int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // 0..m-1
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // 0..k-1
    if (row < m && col < k) {
        float sum = 0.0f;
        for (int l = 0; l < n; ++l) {
            sum += A[row * n + l] * B[col * n + l];
        }
        C[row * k + col] = sum;
    }
}

// ---------------------------------------------------------------------------
// Forward pass. All pointers are DEVICE pointers.
//   d_input  : [batch_size x INPUT_SIZE]
//   d_hidden : [batch_size x HIDDEN_SIZE]  (keeps ReLU output for backward)
//   d_output : [batch_size x OUTPUT_SIZE]  (softmax probabilities)
// ---------------------------------------------------------------------------
void forward(NeuralNetwork *nn, float *d_input, float *d_hidden,
float *d_output, int batch_size) {
    dim3 block(TILE, TILE);

    // Layer 1: hidden = relu(input @ W1 + b1)
    dim3 grid1((HIDDEN_SIZE + TILE - 1) / TILE, (batch_size + TILE - 1) / TILE);
    matmul_a_b_kernel<<<grid1, block>>>(d_input, nn->weights1, d_hidden,
        batch_size, INPUT_SIZE, HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    int n_hidden = batch_size * HIDDEN_SIZE;
    bias_forward_kernel<<<(n_hidden + 255) / 256, 256>>>(d_hidden, nn->bias1,
        batch_size, HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    relu_forward_kernel<<<(n_hidden + 255) / 256, 256>>>(d_hidden, n_hidden);
    CUDA_CHECK(cudaGetLastError());

    // Layer 2: output = softmax(hidden @ W2 + b2)
    dim3 grid2((OUTPUT_SIZE + TILE - 1) / TILE, (batch_size + TILE - 1) / TILE);
    matmul_a_b_kernel<<<grid2, block>>>(d_hidden, nn->weights2, d_output,
        batch_size, HIDDEN_SIZE, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());

    int n_out = batch_size * OUTPUT_SIZE;
    bias_forward_kernel<<<(n_out + 255) / 256, 256>>>(d_output, nn->bias2,
        batch_size, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());

    softmax_kernel<<<batch_size, 1>>>(d_output, batch_size, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());
}

void free_neural_network(NeuralNetwork *nn) {
    CUDA_CHECK(cudaFree(nn->weights1));
    CUDA_CHECK(cudaFree(nn->weights2));
    CUDA_CHECK(cudaFree(nn->bias1));
    CUDA_CHECK(cudaFree(nn->bias2));
    CUDA_CHECK(cudaFree(nn->grad_weights1));
    CUDA_CHECK(cudaFree(nn->grad_weights2));
    CUDA_CHECK(cudaFree(nn->grad_bias1));
    CUDA_CHECK(cudaFree(nn->grad_bias2));
}


