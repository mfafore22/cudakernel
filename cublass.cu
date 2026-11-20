// cuBLAS reference notes (not compiled code).
// The real code using these lives in neural.cu.
//
// cublasSgemm: C = alpha * op(A) @ op(B) + beta * C   (column-major!)
//
// cublasStatus_t cublasSgemm(cublasHandle_t handle,
//                            cublasOperation_t transa,
//                            cublasOperation_t transb,
//                            int m, int n, int k,
//                            const float *alpha,
//                            const float *A, int lda,
//                            const float *B, int ldb,
//                            const float *beta,
//                            float *C, int ldc);
//
// Row-major trick: to get row-major C[M x N] = A[M x K] @ B[K x N], call
//   cublasSgemm(h, N, N,  N, M, K,  &alpha, B, N,  A, K,  &beta, C, N)
// i.e. swap A and B, and swap m/n.
//
// cublasSaxpy: y = alpha * x + y   (used for the SGD weight update)
//
// cublasStatus_t cublasSaxpy(cublasHandle_t handle, int n,
//                            const float *alpha,
//                            const float *x, int incx,
//                            float *y, int incy);
