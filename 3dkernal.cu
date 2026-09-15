__global__ void tensorAdd3D_kernel(
    const float* A,
    const float* B,
    float* C,
    int depth,
    int height,
    int width
){
    int x_dim = blockIdx.x * blockDim.x + threadIdx.x;
    int y_dim = blockIdx.y * blockDim.y + threadIdx.y;
    int z_dim = blockIdx.z * blockDim.z + threadIdx.z;

    if (z_dim < depth && y_dim < height && x_dim < width){
        int index = z_dim * (height * width) + y_dim * width + x_dim;
        C[index] = A[index] + B[index];
    }
}