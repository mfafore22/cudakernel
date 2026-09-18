__global__ void matrixAdd_kernel(const float* A, const float* B, float* C,
int num_rows, int num_cols) {
    int column = blockIdx.x * blockDim.x + threadIdx.x; //
    int row = blockIdx.y * blockDim.y + threadIdx.y;    //

    if (row < num_rows && column < num_cols) { //
        int index = row * num_cols + column;   //
        C[index] = A[index] + B[index];
    }
}