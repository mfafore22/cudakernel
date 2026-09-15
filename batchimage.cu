int batch_size = 6, channels = 3, height = 128, width= 128;

__global__ void normalizeImageBatch(
    const float* input,
    float* output,
    float mean,
    float std,
    int batch_size,
    int channels,
    int height,
    int width
){
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z * blockDim.z + threadIdx.z;

    if (w < width && h < height && c < channels){
        for(int b = 0; b < batch_size; ++b){
            int index = b * (channels * height * width) +
                        c * (height * width) + 
                        h * width + w;
            output[index] = (input[index] - mean) / std;

        }
    }
}