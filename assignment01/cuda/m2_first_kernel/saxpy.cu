#include <stdio.h>

void initVec(float* x, float* y, int n)
{
    for (int i = 0; i < n; i++)
    {
        x[i] = ((i % 2048) - 1024) * 0.5f;
        y[i] = (i % 1024) - 512;
    }
}

__global__ void saxpy(float* x, float* y, int n)
{
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < n; idx += blockDim.x * gridDim.x)
    {
        y[idx] = 2.0f * x[idx] + y[idx];
    }
}


void compute(int n)
{
    float* x = nullptr;
    float* y = nullptr;
    float* dev_x = nullptr;
    float* dev_y = nullptr;
    int bytes = n * sizeof(float);

    cudaMallocHost(&x, bytes);
    cudaMallocHost(&y, bytes);
    cudaMalloc(&dev_x, bytes);
    cudaMalloc(&dev_y, bytes);

    initVec(x, y, n);
    int threads = 1024;
    int blocks = (n + threads - 1) / threads;

    cudaEvent_t start_, stop_;
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
    
    cudaMemcpy(dev_x, x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y, y, bytes, cudaMemcpyHostToDevice);

    cudaEventRecord(start_);
    saxpy<<<blocks, threads>>>(dev_x, dev_y, n);
    cudaEventRecord(stop_);

    cudaMemcpy(y, dev_y, bytes, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop_);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, start_, stop_);
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);

    double s = 0;
    for (int i = 0; i < n; i++)
        s += (double) y[i];

    printf("SUM=%.0f  TIME=%f\n", s, ms);

    cudaFree(dev_x);
    cudaFree(dev_y);
    cudaFreeHost(x);
    cudaFreeHost(y);
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        exit(1);
    }
    int n = std::atoi(argv[1]);
    compute(n);
    return 0;
}