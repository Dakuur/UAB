__global__ void scalar_mult_kernel(float *vect1, float *vect2, float *result, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        result[index] = vect1[index] * vect2[index];
    }
}

__global__ void scalar_mult_kernel(float *vect1, float *vect2, float *result, int n) {
    int index = blockIdx.x * blockDim.x * 4 + threadIdx.x * 4;
    if (index < n) {
        float4 v1 = *((float4*)&vect1[index]);
        float4 v2 = *((float4*)&vect2[index]);
        *((float4*)&result[index]) = v1 * v2;
    }
}


