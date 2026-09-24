__global__ void softmax_naive_kernel(const float* in , float* out, int num_rows , int num_cols){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int column = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < num_rows && column < num_cols){
        float max_val = -1e20f;
        for(int col_idx = 0; col_idx < num_cols; ++col_idx){
            if(in[row * num_cols + col_idx] > max_val) max_val = in[row * num_cols + col_idx];
        }

        float sum_exp = 0.0f;
        for(int col_idx = 0; col_idx < num_cols; ++col_idx){
            sum_exp += expf(in[row * num_cols + col_idx] - max_val);

        }

        out[row * num_cols + column] = expf(in[row * num_cols + column] -
        max_val) / sum_exp;
    }
}