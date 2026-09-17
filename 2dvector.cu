__global__ void vectorAdd_kernel(const flaot* a, const float* b, const float* c, int num_elements){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < num_elements){
        c[index] = a[index] + b[index];
    }
}