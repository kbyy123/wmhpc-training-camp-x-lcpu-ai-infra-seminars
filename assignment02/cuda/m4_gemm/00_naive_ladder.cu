// Reuse assignment01's exact FP32 kernel; replace its 1024^3 CPU harness
// with a 4096^3 cuBLAS correctness check for the requested ladder comparison.
#include "../common.h"
#include <cublas_v2.h>
#include <vector>
constexpr int BS=16;
// Exact body from assignment01/cuda/bonus/matmul.cu.
__global__ void matmul_naive(const float *A, const float *B, float *C,
                             int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.f;
        for (int k = 0; k < K; k++) acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}

int main() {
    constexpr int M=4096,N=4096,K=4096;
    std::vector<float> a((size_t)M*K),b((size_t)K*N),got((size_t)M*N),ref(got.size());
    for(size_t i=0;i<a.size();++i) a[i]=float(int((i*17+3)%7)-3);
    for(size_t i=0;i<b.size();++i) b[i]=float(int((i*13+1)%7)-3);
    float *da,*db,*dc,*dr;
    CUDA_CHECK(cudaMalloc(&da,a.size()*4));CUDA_CHECK(cudaMalloc(&db,b.size()*4));
    CUDA_CHECK(cudaMalloc(&dc,got.size()*4));CUDA_CHECK(cudaMalloc(&dr,ref.size()*4));
    CUDA_CHECK(cudaMemcpy(da,a.data(),a.size()*4,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db,b.data(),b.size()*4,cudaMemcpyHostToDevice));
    auto launch=[&]{matmul_naive<<<dim3(N/BS,M/BS),dim3(BS,BS)>>>(da,db,dc,M,N,K);};
    launch();CUDA_CHECK_KERNEL();
    cublasHandle_t h; if(cublasCreate(&h)!=CUBLAS_STATUS_SUCCESS)return 1;
    float alpha=1,beta=0;
    if(cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&alpha,db,N,da,K,&beta,dr,N)!=CUBLAS_STATUS_SUCCESS)return 1;
    CUDA_CHECK(cudaMemcpy(got.data(),dc,got.size()*4,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ref.data(),dr,ref.size()*4,cudaMemcpyDeviceToHost));
    size_t bad=0;for(size_t i=0;i<got.size();++i)bad+=got[i]!=ref[i];
    GpuTimer timer;timer.start();for(int i=0;i<20;++i)launch();
    const float ms=timer.stop_ms()/20;CUDA_CHECK_KERNEL();
    printf("naive original BS=%d 4096^3 %s bad=%zu %.6f ms %.4f TFLOPS\n",BS,bad?"FAIL":"PASS",bad,ms,2.0*M*N*K/(ms*1e9));
    cublasDestroy(h);cudaFree(da);cudaFree(db);cudaFree(dc);cudaFree(dr);
    return bad!=0;
}
