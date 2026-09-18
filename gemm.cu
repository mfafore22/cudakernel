__global__ void gemm_kernel(const float* A, const float* B , float* C, int M_rows, int N_cols, int K_shared_dim){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int column = blockIdx.x * blockDim.x + threadIdx.x;


    if (row < M_rows && column < N_cols){
        float sum = 0.0f;
        for(int k_idx = 0; k_idx < K_shared_dim; ++k_idx){
            sum += A[row * K_shared_dim + k_idx] * B[k_idx * N_cols + column];
        }
        C[row * N_cols + column] = sum;
        
    }
}