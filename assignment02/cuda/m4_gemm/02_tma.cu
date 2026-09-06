// 问题 4.2(MODIFY):把 4.1 的 staging 换成 TMA,其余不动(仍单缓冲)。
//
// 从你自己的 01_tiled.cu 出发:mma 发射、epilogue、判测口径全部不变,
// 改动集中在两处——host 侧建 tensor map,kernel 侧把 st.shared staging
// 换成 cp.async.bulk.tensor + mbarrier。
//
// 直接告知的事实(工具链与布局配对,不属于考核点):
//   - tensor map 用驱动 API cuTensorMapEncodeTiled 建(Makefile 已链
//     -lcuda);kernel 参数按 const __grid_constant__ CUtensorMap 传
//   - 维度次序:dim0 是最内维(这里是 K,单位为元素数);globalStrides
//     只填外维的字节跨度 {K*2};box 是一次搬运的块 {BK, BM}(B 矩阵
//     {BK, BN});elementStrides 全 1
//   - swizzle 选 CU_TENSOR_MAP_SWIZZLE_128B:TMA 硬件落进 smem 的布局
//     与你 4.1 手工 swz128 摆出来的完全相同,descriptor 一个字段都
//     不用改;interleave/L2 promotion/oob fill 都取 NONE
//   - fence 口径(2.1(b) 在这里兑现):TMA 写 smem 与 tcgen05 读 smem
//     都走 async proxy,fence.proxy.async 不再需要;mbar_wait 之后的
//     tcgen05.fence::after_thread_sync 仍然要
//
// 交付:PASS + 梯子表第二行;回答 handout 4.2 的问题(相对 4.1 的提升
// 为什么这么大——4.1 的 staging 成本由什么构成,用 ncu 佐证)。
//
// 运行:make run/m4_gemm/02_tma;自定形状 ./bin/m4_gemm/02_tma M N K
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <random>
#include <vector>
#include "../common.h"

constexpr int BM = 128, BN = 64, BK = 64;

// SM100 smem descriptor(与 4.1 相同;swz128 已经不需要了)。
__device__ inline uint64_t make_desc_sm100(uint32_t saddr, uint32_t lbo,
                                           uint32_t sbo, uint32_t layout) {
    uint64_t d = 0;
    d |= (uint64_t)((saddr >> 4) & 0x3FFF);
    d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16;
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)layout << 61;
    return d;
}

__device__ inline void mbar_wait(uint32_t mbar, uint32_t phase) {
    uint32_t done = 0;
    while (!done)
        asm volatile(
            "{\n.reg .pred p;\n"
            "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
            "selp.b32 %0, 1, 0, p;\n}"
            : "=r"(done)
            : "r"(mbar), "r"(phase));
}

__device__ __forceinline__ void issue_mma(uint32_t taddr, uint64_t a_desc, uint64_t b_desc, uint32_t idesc, bool accumlate) {
    uint32_t acc = accumlate ? 1 : 0;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(acc));
}

__global__ void gemm_tma(const __nv_bfloat16* gA, const __nv_bfloat16* gB,
                         float* gD, int M, int N, int K,
                         const __grid_constant__ CUtensorMap tmapA,
                         const __grid_constant__ CUtensorMap tmapB) {
    extern __shared__ uint8_t smem_raw[];
    uint8_t* smem =
        (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);

    // TODO:把你 4.1 的 kernel 搬进来,K 循环的 staging 部分改为:
    // (1) 多初始化一组 mbarrier:full(TMA 到达)。4.1 里等 mma 消费
    //     完成的那个继续当 empty 用
    // (2) 每轮:除首轮外先等 empty(smem 可覆写)→ 单线程发 TMA →
    //     等 full → mma(与 4.1 相同)→ commit
    //     发 TMA = 一条 mbarrier.arrive.expect_tx(字节数一次报满
    //     (BM+BN)*BK*2,A、B 两条拷贝共用一个 mbar)+ 两条
    //     cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::
    //     complete_tx::bytes,坐标次序与 tensor map 的维度次序一致:
    //     A 是 {it*BK, tileM},B 是 {it*BK, tileN}
    // (3) 删掉 st.shared staging、swz128、fence.proxy.async(见文件头)
    // full/empty 的 parity 都随轮次翻转,想清楚各自翻转的节奏。
    
    // 一个 block 完成一次 m128n64k64 tile 的乘加，由四条 m128n64k16 的 mma 组成

    // (1) mbarrier 初始化 + TMEM 分配(与 3.2 相同,整段沿用)
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;
    constexpr int TX_BYTES = (BM + BN) * BK * 2;

    __shared__ uint32_t s_taddr;
    __shared__ alignas(8) uint64_t full_bar;
    __shared__ alignas(8) uint64_t empty_bar;
    uint32_t full_addr = (uint32_t)__cvta_generic_to_shared(&full_bar);
    uint32_t empty_addr = (uint32_t)__cvta_generic_to_shared(&empty_bar);

    if (warp_id == 0) { 
        if (lane == 0) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                        :: "r"(full_addr), "r"(1));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                        :: "r"(empty_addr), "r"(1));
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
        uint32_t dst = (uint32_t)__cvta_generic_to_shared(&s_taddr);
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                    :: "r"(dst), "r"(BN));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();

    // (2) 本 block 的输出 tile:tileM = blockIdx.x * BM, tileN = blockIdx.y * BN
    
    int tileM = blockIdx.x * BM, tileN = blockIdx.y * BN;
    
    // (3) K 维循环 it = 0 .. K/BK-1,每轮:

    int full_parity = 0;
    int empty_parity = 0;
    uint32_t tmem_base = s_taddr;
    uint32_t sA_base = __cvta_generic_to_shared(smem);
    uint32_t sB_base = __cvta_generic_to_shared(smem + 2 * BM * BK);

    bool acc = false;
    for (int it = 0; it < K / BK; it++) {
        
    //     prologue：除了第一次以外，先等待 empty 
        if (it != 0) {
            mbar_wait(empty_addr, empty_parity);
            empty_parity ^= 1;
        }
    
    //     (a) 一个线程登记总字节数，发起两次 TMA 
        if (tid == 0) {
            asm volatile(
                "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                :: "r"(full_addr), "r"(TX_BYTES));
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
                "[%0], [%1, {%2, %3}], [%4];"
                :: "r"(sA_base), "l"(&tmapA), "r"(it * BK), "r"(tileM), "r"(full_addr)
            );
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
                "[%0], [%1, {%2, %3}], [%4];"
                :: "r"(sB_base), "l"(&tmapB), "r"(it * BK), "r"(tileN), "r"(full_addr)
            );
        }

    //     (b) TMA 搬运和 mma 都属于 async proxy，可以不用 fence.proxy.async
        __syncthreads();
        mbar_wait(full_addr, full_parity);
        full_parity ^= 1;
        
    //     (c) 单线程发射 4 条 k16 的 tcgen05.mma。注意累加位:整个 K
    //         循环里只有第一条 mma 不累加(enable-input-d = 0),其余
    //         全部累加到同一块 TMEM——3.2 里"kk>0 才累加"的条件在这里
    //         要连 it 一起考虑
        
        constexpr uint32_t idesc = (1u << 4) | (1u << 7) | (1u << 10) | ((BN >> 3) << 17) | ((BM >> 4) << 24);

        if (tid == 0) {
            asm volatile("tcgen05.fence::after_thread_sync;");
            for (int k = 0; k < 4; k++) {
                uint64_t a_desc = make_desc_sm100(sA_base + 32 * k, 0u, 1024u, 2u);
                uint64_t b_desc = make_desc_sm100(sB_base + 32 * k, 0u, 1024u, 2u);
                issue_mma(tmem_base, a_desc, b_desc, idesc, acc);
                acc = true;
            }
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one"
                        ".shared::cluster.b64 [%0];" :: "r"(empty_addr));
        }

    }

    // epilogue 在最后，等待一次 empty
    mbar_wait(empty_addr, empty_parity);
    empty_parity ^= 1;

    // (4) epilogue 与 3.2 相同,写回 gD 的 (tileM, tileN) 块(行跨度 N)
    asm volatile("tcgen05.fence::after_thread_sync;");
    // 总共要搬运 128 * 64 fp32，每个 warp 搬运 32 * 64，单个 thread 搬运 N 个，搬到寄存器
    float t[BN];

    #pragma unroll
    for (int c = 0; c < BN; c += 8) {
        uint32_t taddr = tmem_base + ((warp_id * 32) << 16) + c;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 "
                     "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                    : "=f"(t[c]), "=f"(t[c + 1]), "=f"(t[c + 2]), "=f"(t[c + 3]), 
                      "=f"(t[c + 4]), "=f"(t[c + 5]), "=f"(t[c + 6]), "=f"(t[c + 7])
                    : "r"(taddr));
        asm volatile("tcgen05.wait::ld.sync.aligned;");
    }

    // 每一个 thread 把自己的 N 个 fp32 写回 GMEM
    #pragma unroll
    for (int i = 0; i < BN; i++) {
        int r = warp_id * 32 + lane; 
        gD[(tileM + r) * N + tileN + i] = t[i];
    }

    // (5) dealloc
    __syncthreads();
    if (warp_id == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     :: "r"(tmem_base), "r"(BN));
    }
}

int main(int argc, char** argv) {
    int M = argc > 3 ? atoi(argv[1]) : 4096;
    int N = argc > 3 ? atoi(argv[2]) : 4096;
    int K = argc > 3 ? atoi(argv[3]) : 4096;
    if (M % BM || N % BN || K % BK) {
        printf("形状需按 %dx%dx%d 对齐\n", BM, BN, BK);
        return 1;
    }
    size_t nA = (size_t)M * K, nB = (size_t)N * K, nD = (size_t)M * N;
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-3, 3);
    std::vector<__nv_bfloat16> hA(nA), hB(nB);
    for (auto& v : hA) v = __float2bfloat16((float)dist(rng));
    for (auto& v : hB) v = __float2bfloat16((float)dist(rng));
    __nv_bfloat16 *dA, *dB;
    float *dD, *dRef;
    CUDA_CHECK(cudaMalloc(&dA, nA * 2));
    CUDA_CHECK(cudaMalloc(&dB, nB * 2));
    CUDA_CHECK(cudaMalloc(&dD, nD * 4));
    CUDA_CHECK(cudaMalloc(&dRef, nD * 4));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), nA * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), nB * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dD, 0xFF, nD * 4));

    // TODO:cuTensorMapEncodeTiled 建 tmapA/tmapB(参数要点见文件头;
    // 返回值要检查,CUDA_SUCCESS 之外一律报错退出——tensor map 参数错
    // 的典型症状是 kernel 静默读到 0 或越界,而不是启动失败)。

    // 需要传入的是数组的地址
    uint64_t dimsA[2] = {static_cast<uint64_t>(K), static_cast<uint64_t>(M)};
    uint64_t dimsB[2] = {static_cast<uint64_t>(K), static_cast<uint64_t>(N)};
    uint64_t strides[1] = {static_cast<uint64_t>(K) * 2};
    uint32_t boxA[2] = {BK, BM}, boxB[2] = {BK, BN};
    uint32_t elementStrides[2] = {1, 1};

    CUtensorMap tmapA = {}, tmapB = {};
    // 文档见 https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY_1ga7c7d2aaac9e49294304e755e6f341d7
    CUresult resultA = cuTensorMapEncodeTiled(&tmapA, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 
        2, 
        dA,
        dimsA, 
        strides, 
        boxA, 
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    CUresult resultB = cuTensorMapEncodeTiled(&tmapB, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 
        2, 
        dB,
        dimsB, 
        strides, 
        boxB, 
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    if (resultA != CUDA_SUCCESS || resultB != CUDA_SUCCESS) {
        fprintf(stderr, "Tensor map encode failed: A=%d, B=%d\n",
                static_cast<int>(resultA),
                static_cast<int>(resultB));
        return 1;
    }
    dim3 grid(M / BM, N / BN);
    size_t smemBytes = (size_t)(BM + BN) * BK * 2 + 1024;
    CUDA_CHECK(cudaFuncSetAttribute(gemm_tma,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)smemBytes));
    auto launch = [&] {
        gemm_tma<<<grid, 128, smemBytes>>>(dA, dB, dD, M, N, K, tmapA, tmapB);
    };
    launch();
    CUDA_CHECK_KERNEL();

    cublasHandle_t h;
    cublasCreate(&h);
    float alpha = 1.f, beta = 0.f;
    cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, dB, CUDA_R_16BF,
                 K, dA, CUDA_R_16BF, K, &beta, dRef, CUDA_R_32F, N,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> got(nD), ref(nD);
    CUDA_CHECK(cudaMemcpy(got.data(), dD, nD * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ref.data(), dRef, nD * 4, cudaMemcpyDeviceToHost));
    long bad = 0;
    for (size_t i = 0; i < nD; i++) bad += got[i] != ref[i];

    int iters = (size_t)M * N >= (size_t)4096 * 4096 ? 20 : 100;
    float ms = time_avg_ms(launch, iters);
    double tflops = 2.0 * M * N * K / (ms * 1e9);
    float cub_ms = time_avg_ms(
        [&] {
            cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, dB,
                         CUDA_R_16BF, K, dA, CUDA_R_16BF, K, &beta, dRef,
                         CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);
        },
        iters);
    double cub_tflops = 2.0 * M * N * K / (cub_ms * 1e9);
    printf("[4.2 tma] M=%d N=%d K=%d  %s(bad=%ld)  %.2f ms  %.1f TFLOPS  "
           "(cuBLAS %.1f, 达成率 %.0f%%)\n",
           M, N, K, bad ? "FAIL" : "PASS", bad, ms, tflops, cub_tflops,
           100.0 * tflops / cub_tflops);
    cublasDestroy(h);
    return bad != 0;
}
