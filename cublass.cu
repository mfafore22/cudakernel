cublasStatus_t cublasSgemm(cublasHandle_t handle,
                           cublasOperation_t transa,
                           cublasOperation_t transb,
                           int m, int n, int k,
                           const float *alpha,
                           const float *A, int lda,
                           const float *B, int ldb,
                           const float *beta,
                           float *C, int ldc);


                           void update_weights_only(NeuralNetworkCUDA *nn, float lr) {
    float neg_lr = -lr;

    CUBLAS_CHECK(cublasSaxpy(nn->cublas_handle, INPUT_SIZE * HIDDEN_SIZE,
                           &neg_lr, nn->d_grad_weights1, 1,
                           nn->d_weights1, 1));
    CUBLAS_CHECK(cublasSaxpy(nn->cublas_handle, HIDDEN_SIZE * OUTPUT_SIZE,
                           &neg_lr, nn->d_grad_weights2, 1,
                           nn->d_weights2, 1));
    CUBLAS_CHECK(cublasSaxpy(nn->cublas_handle, HIDDEN_SIZE,
                           &neg_lr, nn->d_grad_bias1, 1,
                           nn->d_bias1, 1));
    CUBLAS_CHECK(cublasSaxpy(nn->cublas_handle, OUTPUT_SIZE,
                           &neg_lr, nn->d_grad_bias2, 1,
                           nn->d_bias2, 1));

    CUDA_CHECK(cudaDeviceSynchronize());
}