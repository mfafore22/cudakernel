__global__ void maxpool2d_kernel(const float* in, float* out, int height, int width, int pool_dim ){
  
    int output_col = blockIdx.x * blockDim.x + threadIdx.x;
    int output_row = blockIdx.y * blockDim.y + threadIdx.y;


    int output_h = height / pool_dim;
    int output_w = width / pool_dim;

    if(output_row < output_h && output_col > output_w){
        float max_val = 1e20f;
        for(int pool_row = 0; pool_row < pool_dim; ++pool_row){
            for(int pool_col = 0; pool_col < pool_dim; ++pool_col){
                int input_row = output_row * pool_dim + pool_row;
                int input_col = output_col * pool_dim + pool_Col;
                float val = in[input_row * width + input_col];
                if (val > max_val) max_val = val;
            }
        }
        out[output_row * output_w + output_col] = max_val;
    }
}