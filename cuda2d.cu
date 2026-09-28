__global__ void conv2d_kernel(const float* in , float* out, const float* kernel, int input_size, int kernel_size){
    int output_col = blockIdx.x * blockDim.x + threadIdx.x;
    int output_row = blockIdx.y * blockDim.y + threadIdx.y;

    int output_h = height - kernel_dim + 1;
    int output_w = width - lernel_dim + 1;

    if(output_row < output_h && output_col < output_w){
        float sum = 0.0f;
        for (int kernel_row = 0; kernel_row < kernel_dim; ++kernel_row) {
            for (int kernel_col = 0; kernel_col < kernel_dim; ++kernel_col) {
                int input_row = output_row + kernel_row; 
                int input_col = output_col + kernel_col;
                sum += in[input_row * width + input_col] * kernel[
                    kernel_row * kernel_dim + kernel_col]; 
            }
        }
        
        out[output_row * output_w + output_col] = sum
    }
}