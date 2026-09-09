// 问题 4.3(FROM-SCRATCH,模块压轴):多级缓冲流水。
//
// 从你自己的 02_tma.cu 出发,把单缓冲扩成 STAGES 级循环缓冲:TMA 往
// 前预取后续 K 段,mma 消费当前段,装载与计算重叠。STAGES 是编译参数:
//   STAGES=4 make -B run/m4_gemm/03_pipeline
// (-B 不能省:只改 -D 不改文件,make 会认为无需重编。)
//
// 明确不要求:warp specialization、persistent kernel、epilogue 融合。
// 不设达成率门槛,评分看实验与归因质量。
//
// 两个已知事实,直接告知:
//   1. smem 用量 = STAGES*(BM+BN)*BK*2,STAGES>=3 起超过 48KB 静态
//      上限,必须动态 smem + cudaFuncSetAttribute(main 已配好)。
//   2. 一条真实的流水线 hazard(我们开发答案时踩到的,写出来让你避开):
//      "机会式预取"(try_wait 非阻塞,空了就发)不能替代"强制发射"。
//      若本轮要消费的那段 TMA 在早先检查时 stage 未空而被跳过,后面
//      wait full 等的就是一条从未发出的拷贝——死锁。症状签名很典型:
//      1024^3 侥幸全过,4096^3 必挂(13 万次机会必中一次)。正确结构:
//      本轮要消费的 TMA 用阻塞等 empty 保证发出,机会式 try_wait 只
//      用于更深的预取。另外 empty mbarrier 必须每 stage 一个:单个
//      mbar 的 parity 区分不了相隔 2 轮的完成,STAGES>=2 必然歧义。
//
// 交付:
//   - 梯子表第三行(4096^3,默认 STAGES=3)
//   - stages 扫描表:S ∈ {2,3,4,6},在两个形状上各扫一遍——4096^3 与
//     M=256 N=4096 K=16384(小 grid、长 K)。两张表的 S 敏感度不一样,
//     解释差异来自什么(提示方向:每 SM 常驻 block 数怎么随 smem 用量
//     变、块间并发本身能隐藏多少延迟)。./sweep_stages.sh 会跑全表
//   - 流水时空图:任选一个 S,画出稳态下 TMA/mma 在各 stage 上的重叠
//   - handout 4.3 的三问:瓶颈移动;梯子表逐级归因(含 assignment01
//     的 naive matmul 同口径对照);smem 与 TMEM 谁先顶住扩 stage/tile
//
// 运行:make run/m4_gemm/03_pipeline;./bin/m4_gemm/03_pipeline M N K
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <random>
#include <vector>
#include "../common.h"

#ifndef STAGES
#define STAGES 3
#endif

constexpr int BM = 128, BN = 64, BK = 64;
constexpr int NSTAGE = STAGES;
constexpr int TX_BYTES = (BM + BN) * BK * 2; 

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

// 非阻塞版:成功返回 true。机会式深预取用它。
__device__ inline bool mbar_try(uint32_t mbar, uint32_t phase) {
    uint32_t done;
    asm volatile(
        "{\n.reg .pred p;\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
        "selp.b32 %0, 1, 0, p;\n}"
        : "=r"(done)
        : "r"(mbar), "r"(phase));
    return done;
}

__device__ __forceinline__ void issue_tma(uint32_t full_bar_addr, int it, 
                                          int tileM, int tileN, uint8_t* smem,
                                          const CUtensorMap* tmapA_addr,
                                          const CUtensorMap* tmapB_addr) {
    int stage = it % NSTAGE;

    uint32_t sA_base = __cvta_generic_to_shared(smem + stage * TX_BYTES);
    uint32_t sB_base = __cvta_generic_to_shared(smem + stage * TX_BYTES + 2 * BM * BK);
                                    
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"(full_bar_addr), "r"(TX_BYTES));
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];"
        :: "r"(sA_base), "l"(tmapA_addr), "r"(it * BK), "r"(tileM), "r"(full_bar_addr)
    );
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];"
        :: "r"(sB_base), "l"(tmapB_addr), "r"(it * BK), "r"(tileN), "r"(full_bar_addr)
    );
                                        
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

__global__ void gemm_pipeline(const __nv_bfloat16* gA, const __nv_bfloat16* gB,
                              float* gD, int M, int N, int K,
                              const __grid_constant__ CUtensorMap tmapA,
                              const __grid_constant__ CUtensorMap tmapB) {
    extern __shared__ uint8_t smem_raw[];
    uint8_t* smem =
        (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);

    // TODO:把你 4.2 的 kernel 扩成 NSTAGE 级流水。参考结构:
    // (1) smem 划成 NSTAGE 段,stage s 的 A/B 起点自己排;mbarrier 每
    //     stage 两个:full[s](TMA 到达)、empty[s](mma 消费完成)
    // (2) 预热:先发 min(NSTAGE, iters) 轮 TMA(发第 it 轮 = 对 stage
    //     it%NSTAGE 做 arrive.expect_tx + 两条 cp.async.bulk.tensor)
    // (3) 主循环 it:
    //     - 强制发射:若第 it 轮 TMA 还没发,阻塞等 empty[it%NSTAGE]
    //       后补发(见文件头 hazard;empty 的 parity 按该 stage 被复用
    //       的轮次算,第一次复用等的是上一轮使用的完成)
    //     - 机会式深预取:try_wait 下一个待发 stage 的 empty,成功就
    //       继续发,失败立刻停,不许阻塞
    //     - 等 full[it%NSTAGE](parity = (it/NSTAGE)&1)→ tcgen05.fence
    //       → mma(与 4.2 相同,累加位口径不变)→ commit 到
    //       empty[it%NSTAGE]
    // (4) drain:等最后一轮 mma 的 empty 到达,再进 epilogue

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;
    int tileM = blockIdx.x * BM, tileN = blockIdx.y * BN;
    
    // next 表示下一段尚未发起搬运的数据
    int next = 0;

    __shared__ uint32_t s_taddr;
    __shared__ alignas(8) uint64_t full_bar[NSTAGE];
    __shared__ alignas(8) uint64_t empty_bar[NSTAGE];
    uint32_t full_bar_addr = (uint32_t)__cvta_generic_to_shared(full_bar);
    uint32_t empty_bar_addr = (uint32_t)__cvta_generic_to_shared(empty_bar);

    // 初始化所有的 mbarrier
    if (warp_id == 0) { 
        if (lane == 0) {
            for (int i = 0; i < NSTAGE; i++) {
                asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                            :: "r"(full_bar_addr + 8 * i), "r"(1));
                asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                            :: "r"(empty_bar_addr + 8 * i), "r"(1));
                asm volatile("fence.mbarrier_init.release.cluster;");
            }
        }
        uint32_t dst = (uint32_t)__cvta_generic_to_shared(&s_taddr);
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                    :: "r"(dst), "r"(BN));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();

    // TMEM 的地址，放入寄存器中避免多次访问 SMEM
    uint32_t tmem_base = s_taddr;

    bool acc = false;
    int max_iter = K / BK;

    // 预热：先搬运 NSTAGE 组数据
// __device__ __forceinline__ void issue_tma(uint32_t full_bar_addr, int it, 
//                                           int tileM, int tileN, uint8_t* smem,
//                                           const CUtensorMap* tmapA_addr,
//                                           const CUtensorMap* tmapB_addr);
    
    if (tid == 0) {
        for (next = 0; next < min(max_iter, NSTAGE); next++) {
                issue_tma(full_bar_addr + 8 * next, next, tileM, tileN, smem, &tmapA, &tmapB);
        }
        
        // generation = 0 的 tma 预处理；mma 此时不需要等待 empty，需要等到 full = 0 的 parity
        // generation = 1 的 tma 需要等到 (generation - 1) & 1 = 0 的 empty parity；mma 需要等到 generation & 1 = 1 的 full
        for (int it = 0; it < max_iter; it++) {
            
            int stage = it % NSTAGE;
            int generation = it / NSTAGE;

            // 若下一个还没搬运数据的是当前 it，不能跳过
            if (next == it) {
                // 搬之前需要等到上一个 generation 的 mma 计算完成
                if (generation) {
                    mbar_wait(empty_bar_addr + 8 * stage, (generation - 1) & 1);
                }
                issue_tma(full_bar_addr + 8 * stage, next, tileM, tileN, smem, &tmapA, &tmapB);
                next++;
            } 

            // 尝试提前发起后面的 tma，但是不能套圈
            for (; next < max_iter && next < it + NSTAGE; next++) {
                int next_stage = next % NSTAGE;
                int next_gene = next / NSTAGE;
                // 等待下一段尚未发起搬运的数据对应的 mma 是否完毕，如果未完毕，直接结束搬运
                if (!mbar_try(empty_bar_addr + 8 * next_stage, (next_gene - 1) & 1)) {
                    break;
                }
                // 如果已完毕，就发起 tma，并继续循环
                issue_tma(full_bar_addr + 8 * next_stage, next, tileM, tileN, smem, &tmapA, &tmapB);
            }

            // mma 计算当前 it
            mbar_wait(full_bar_addr + 8 * stage, generation & 1);
            constexpr uint32_t idesc = (1u << 4) | (1u << 7) | (1u << 10) | ((BN >> 3) << 17) | ((BM >> 4) << 24);
            
            uint32_t sA_base = __cvta_generic_to_shared(smem + stage * TX_BYTES);
            uint32_t sB_base = __cvta_generic_to_shared(smem + stage * TX_BYTES + 2 * BM * BK);
            
            if (tid == 0) {
                asm volatile("tcgen05.fence::after_thread_sync;");
                for (int k = 0; k < 4; k++) {
                    
                    uint64_t a_desc = make_desc_sm100(sA_base + 32 * k, 0u, 1024u, 2u);
                    uint64_t b_desc = make_desc_sm100(sB_base + 32 * k, 0u, 1024u, 2u);
                    issue_mma(tmem_base, a_desc, b_desc, idesc, acc);
                    acc = true;
                }
                asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one"
                            ".shared::cluster.b64 [%0];" :: "r"(empty_bar_addr + 8 * stage));
            }
        }
    }

    __syncthreads();
    // 所有 mma 都发射完，等待最后一次
    int last = max_iter - 1;
    mbar_wait(empty_bar_addr + 8 * (last % NSTAGE), (last / NSTAGE) & 1);
   
    // epilogue 与之前相同,写回 gD 的 (tileM, tileN) 块(行跨度 N)
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

    // TODO:tensor map 从你的 4.2 原样复制。
    CUtensorMap tmapA = {}, tmapB = {};

    uint64_t dimsA[2] = {static_cast<uint64_t>(K), static_cast<uint64_t>(M)};
    uint64_t dimsB[2] = {static_cast<uint64_t>(K), static_cast<uint64_t>(N)};
    uint64_t strides[1] = {static_cast<uint64_t>(K) * 2};
    uint32_t boxA[2] = {BK, BM}, boxB[2] = {BK, BN};
    uint32_t elementStrides[2] = {1, 1};
    
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
    // NSTAGE=3 时 72KB+对齐余量,超 48KB 静态上限,动态 smem 必须。
    size_t smemBytes = (size_t)NSTAGE * (BM + BN) * BK * 2 + 1024;
    CUDA_CHECK(cudaFuncSetAttribute(gemm_pipeline,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)smemBytes));
    auto launch = [&] {
        gemm_pipeline<<<grid, 128, smemBytes>>>(dA, dB, dD, M, N, K, tmapA,
                                                tmapB);
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
    printf("[4.3 pipeline S=%d] M=%d N=%d K=%d  %s(bad=%ld)  %.2f ms  %.1f "
           "TFLOPS  (cuBLAS %.1f, 达成率 %.0f%%)\n",
           NSTAGE, M, N, K, bad ? "FAIL" : "PASS", bad, ms, tflops,
           cub_tflops, 100.0 * tflops / cub_tflops);
    cublasDestroy(h);
    return bad != 0;
}
