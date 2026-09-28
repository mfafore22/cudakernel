__global__ void conv1d_kernel(const float* in , float* out, const float* kernel, int input_size, int kernel_size){
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int output_szie = input_size - kernel_szie + 1;

    if(output_idx < output_size){
        float sum = 0.0f;
        for(int k_idx = 0; k_idx < kernel_size; ++k_idx){
            sum += in[output_idx + k_idx] * kernel[k_idx];

        }
        out[output_idx] = sum;
    }
}