__global__ void vectorAdd_kernel(const float* a, const float* b, float* c, int num_elements){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < num_elements){
        c[index] = a[index] + b[index];
    }
}