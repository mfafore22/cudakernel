void tensorAdd3D_cpu(
    const float* A,
    const float* B,
    float* C,
    int depth,
    int height,
    int width
){
    for (int d = 0; d < depth; ++d){
        for (int h = 0; h < height; ++h){
            for (int w = 0; w < width; ++w){
                int index = d * (height * width) + h * width + w;
                C[index] = A[index] + B[index];
            }
        }
    }
}