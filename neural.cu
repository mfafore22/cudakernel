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