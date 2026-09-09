#define main pipeline_original_main
#include "03_pipeline.cu"
#undef main
int main() {
    cudaDeviceProp p;CUDA_CHECK(cudaGetDeviceProperties(&p,0));
    cudaFuncAttributes a;CUDA_CHECK(cudaFuncGetAttributes(&a,gemm_pipeline));
    const size_t bytes=NSTAGE*STAGE_BYTES+1024;
    CUDA_CHECK(cudaFuncSetAttribute(gemm_pipeline,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
    int resident=0;CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,gemm_pipeline,128,bytes));
    printf("GPU=%s CC=%d.%d SMs=%d shared_per_SM=%zu shared_optin=%zu L2=%d\n",p.name,p.major,p.minor,p.multiProcessorCount,p.sharedMemPerMultiprocessor,p.sharedMemPerBlockOptin,p.l2CacheSize);
    printf("S=%d registers=%d local_bytes=%zu static_smem=%zu dynamic_smem=%zu maxthreads=%d resident=%d\n",NSTAGE,a.numRegs,a.localSizeBytes,a.sharedSizeBytes,bytes,a.maxThreadsPerBlock,resident);
    CUDA_CHECK(cudaFuncSetAttribute(gemm_pipeline,cudaFuncAttributePreferredSharedMemoryCarveout,100));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,gemm_pipeline,128,bytes));
    printf("with_max_shared_carveout resident=%d\n",resident);
}
