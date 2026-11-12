#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>   // for memset

#define INPUT_SIZE 784
#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 10
#define BATCH_SIZE 8
#define EPOCHS 10
#define LEARNING_RATE 0.01
#define TRAIN_SIZE 10000
#define TEST_SIZE 10000


typedef struct {
    float *weights1, *weights2, *bias1, *bias2;
    float *grad_weights1, *grad_weights2, *grad_bias1, *grad_bias2;
} NeuralNetwork;


void initialize_weights(float *weights, int input_size, int output_size) {
    float scale = sqrtf(6.0f / input_size);
    for (int i = 0; i < input_size * output_size; i++) {
        weights[i] = ((float)rand() / RAND_MAX) *
        2.0f * scale - scale;
    }
}

void initialize_bias(float *bias, int size) {
    for (int i = 0; i < size; i++) {
        bias[i] = 0.0f;
    }
}

void initialize_neural_network(NeuralNetwork *nn) {
    nn->weights1 = malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    nn->weights2 = malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    nn->bias1 = malloc(HIDDEN_SIZE * sizeof(float));
    nn->bias2 = malloc(OUTPUT_SIZE * sizeof(float));
    nn->grad_weights1 = malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    nn->grad_weights2 = malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    nn->grad_bias1 = malloc(HIDDEN_SIZE * sizeof(float));
    nn->grad_bias2 = malloc(OUTPUT_SIZE * sizeof(float));

    initialize_weights(nn->weights1, INPUT_SIZE, HIDDEN_SIZE);
    initialize_weights(nn->weights2, HIDDEN_SIZE, OUTPUT_SIZE);
    initialize_bias(nn->bias1, HIDDEN_SIZE);
    initialize_bias(nn->bias2, OUTPUT_SIZE);
}


void matmul_a_b(float *A, float *B, float *C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            C[i * k + j] = 0.0f;
            for (int l = 0; l < n; l++) {
                C[i * k + j] += A[i * n + l] * B[l * k + j];
            }
        }
    }
}

void bias_forward(float *x, float *bias, int batch_size, int size) {
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < size; i++) {
            x[b * size + i] += bias[i];
        }
    }
}

void relu_forward(float *x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = fmaxf(0.0f, x[i]);
    }
}