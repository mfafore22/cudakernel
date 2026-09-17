void matrixAdd_cpu(const float* A, const float* B, float* C, int num_rows, int num_cols){
    for(int row = 0; row < num_rows; ++row){
        for(int col = 0; col < num_cols; ++col){
            for(int col = 0; col < num_cols; ++col){
                int index = row * num_cols + col;
                C[index] = A[index] + B[index];
            }
        }
    }
}