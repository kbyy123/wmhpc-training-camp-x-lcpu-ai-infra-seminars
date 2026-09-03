// 问题 3.2(模块压轴):从零写 tcgen05 单 tile GEMM。
//
// 形状 m128n64k64,bf16 输入,f32 累加,cta_group::1,单 block 128 线程。
// 数据通路:global -> smem(K-major + 128B swizzle)-> tcgen05.mma ->
// TMEM -> tcgen05.ld -> global。判测(main 已给出)用小整数严格对拍。
//
// 给你的材料:课件 F27 的七步流程(下面 kernel 里只留了步骤注释)、
// 你在 2.2 写的 descriptor 编码(SM100 位域)、2.3 的 swizzle_128B
// (staging 布局用它;布局错,结果必错——这里是它的真硬件判测)。
// 其余(TMEM alloc、mbarrier、idesc、tcgen05.mma/ld 的写法)自己查
// PTX ISA 对应章节,课件 C15-C21 讲过每一件的语义,数字换成本题形状。
//
// 两个提醒,直接说明:
// - smem 写完到发射 mma 之间需要 fence.proxy.async(2.1 排序题的答案
//   在这里上真硬件;漏掉的现象自己观察一次,写进报告)
// - tcgen05.ld 每个 warp 只能读自己的 32 条 lane(3.1(a));taddr 高
//   16 bit 是 lane 偏移、低 16 bit 是列偏移;ld 之后要 tcgen05.wait::ld
//
// 运行:make run/m3_tcgen05/02_single_tile;多 seed:./judge_tile.sh
#include <cuda_bf16.h>
#include <cstdio>
#include <random>
#include "../common.h"

constexpr int M = 128, N = 64, K = 64;

// 128B swizzle 的物理偏移(即 2.3 的 swizzle_128B;row 是 K-major 下的
// 行 = M 或 N 维,col 是 K 维字节)。atom = 8 行 × 128B,SBO=1024。
__host__ __device__ inline int swz128(int row, int colByte) {
    int atom = row >> 3, r = row & 7, chunk = colByte >> 4, in16 = colByte & 15;
    return atom * 1024 + r * 128 + ((chunk ^ r) << 4) + in16;
}

__device__ inline uint64_t make_desc_sm100(uint32_t saddr, uint32_t lbo,
                                           uint32_t sbo, uint32_t layout) {
    uint64_t d = 0;
    d |= (uint64_t)((saddr >> 4) & 0x3FFF);
    d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16;
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;             // version = 1(SM100)
    d |= (uint64_t)layout << 61;        // 3 bit layout type
    return d;
}

__device__ inline void mbar_wait(uint32_t addr_bar, uint32_t phase) {
    uint32_t done = 0;
    while (!done)
        asm volatile(
            "{\n.reg .pred p;\n"
            "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
            "selp.b32 %0, 1, 0, p;\n}"
            : "=r"(done)
            : "r"(addr_bar), "r"(phase));
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

__global__ void tcgen05_tile(const __nv_bfloat16* gA, const __nv_bfloat16* gB,
                             float* gD) {
    // TODO: 按七步实现。
    
    // (1) mbarrier 初始化 + TMEM 分配(alloc 结果写到 shared,广播)
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;

    // 分配的 TMEM 内存地址回写回到 SMEM 里，这里 s_taddr 可以看作指针
    __shared__ uint32_t s_taddr;
    // mbarrier 对象，要求 8B 对齐
    __shared__ alignas(8) uint64_t bar;
    // addr_bar 是 mbarrier 对象在 SMEM 下的地址 
    uint32_t addr_bar = (uint32_t)__cvta_generic_to_shared(&bar);

    // 分配 TMEM 空间需要整个 warp
    if (warp_id == 0) { 
        // init mbarrier 只需要一个 thread
        // %1 为 init 1 次 arrive count
        // .release.cluster 把作用域推广到 cluster（此处退化成 CTA，不影响）
        if (lane == 0) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                        :: "r"(addr_bar), "r"(1));
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
        uint32_t dst = (uint32_t)__cvta_generic_to_shared(&s_taddr);
        // 分配 TMEM 地址是 128 行一起分配，N 是分类列数，刚好对应 M * N 
        // 分配的地址返回到 SMEM 的指针 [%0] dst 
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                    :: "r"(dst), "r"(N));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();

    // (2) 全体线程把 A/B 按 swizzled 布局写进 smem

    // 128B swizzled 布局，8 行 * 128B = 1024B，需要对齐
    __shared__ alignas(1024)  __nv_bfloat16 sA[M * K], sB[N * K];
    for (int i = tid; i < M * K; i += blockDim.x) {
        int r = i / K, c = i % K;
        // c * 2 是当前行的字节偏移（一个 bf16 占 2 字节）
        // swz128(r, c * 2) 是距离数组起始的字节偏移（将字节转化回 bf16 下标）
        sA[swz128(r, c * 2) / 2] = gA[i];
    }
    for (int i = tid; i < N * K; i += blockDim.x) {
        int r = i / K, c = i % K;
        sB[swz128(r, c * 2) / 2] = gB[i];
    }

    // (3) fence.proxy.async + __syncthreads

    asm volatile("fence.proxy.async.shared::cta;");
    // 即将发射 mma，需要让 async proxy 读取到 generic proxy 存进 SMEM的矩阵
    __syncthreads();

    // (4) 单线程发射 4 条 k16 的 tcgen05.mma(第一条不累加),commit

    // 取出分配的 TMEM 地址指针，用于发射 mma
    uint32_t tmem_base = s_taddr;
    // 以 SMEM 格式地址来放入 descriptor
    uint32_t sA_base = __cvta_generic_to_shared(sA);
    uint32_t sB_base = __cvta_generic_to_shared(sB);

    // 该条指令的 idesc，4 次 mma 都不变
    constexpr uint32_t idesc = (1u << 4) | (1u << 7) | (1u << 10) | ((N >> 3) << 17) | ((M >> 4) << 24);

    // dense tcgen05.mma.cta_group::1.kind::f16 的单条指令固定 k = 16，因此分为 4 个 slice
    if (tid == 0) {
        // 跨线程场景下，线程同步之后、执行自己的tcgen05 指令之前需要该指令
        asm volatile("tcgen05.fence::after_thread_sync;");

        // 第一发 mma 不开启累加，后续三发开启
        bool acc = false;
        for (int k = 0; k < 4; k++) {
            // 对 A：一次处理 128 * 16 tile，起始地址每次偏移 16 * 2B = 32B
            // 根据 128B swizzle，一行 128B 连续存储（A 一行恰好 128B），因此 A 可以看作常规数组，完全连续存储（不考虑 swizzle）
            // 但是由于 descriptor 有 LBO 和 SBO，mma 可以知道 A 矩阵的逻辑形状
            // 128B swizzle 下 LBO = 0，SBO = 1024B（见 m2_smem/02_descriptor.cu）
            // 128B swizzle 对应 layout = 2
            // 注意矩阵的 desc 为 64 bit，mma 的 idesc 为 32 bit
            uint64_t a_desc = make_desc_sm100(sA_base + 32 * k, 0u, 1024u, 2u);
            uint64_t b_desc = make_desc_sm100(sB_base + 32 * k, 0u, 1024u, 2u);
            issue_mma(tmem_base, a_desc, b_desc, idesc, acc);
            acc = true;
        }
        // 线程发射完毕后 commit，跟踪当前线程此前发射的所有异步 tcgen05 操作
        // 等待完成后进行 arrive 一次，使得 arrival count = 0，phase 翻转
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one"
                     ".shared::cluster.b64 [%0];" :: "r"(addr_bar));
    }

    // (5) mbarrier 等待

    // 所有线程等待 mbarrier 完成
    mbar_wait(addr_bar, 0);

    // (6) epilogue:每 warp tcgen05.ld 自己的 32 条 lane,写回 global 

    // 跨线程场景下，线程同步之后、执行自己的tcgen05 指令之前需要该指令
    asm volatile("tcgen05.fence::after_thread_sync;");
    // 总共要搬运 128 * 64 fp32，每个 warp 搬运 32 * 64，单个 thread 搬运 N 个，搬到寄存器
    float t[N];

    // 进行 8 次 tcgen05.ld，每次 tile 为 128 * 8
    #pragma unroll
    for (int c = 0; c < N; c += 8) {
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
    for (int i = 0; i < N; i++) {
        int row = warp_id * 32 + lane; 
        gD[row * N + i] = t[i];
    }

    // (7) __syncthreads 后 dealloc

    // sync 保证不会出现有的 warp 还在写，warp0 就把 TMEM 释放的情况
    __syncthreads();
    if (warp_id == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     :: "r"(tmem_base), "r"(N));
    }
}

int main(int argc, char** argv) {
    unsigned seed = argc > 1 ? (unsigned)atoi(argv[1]) : 42;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(-3, 3);
    std::vector<__nv_bfloat16> hA(M * K), hB(N * K);
    std::vector<float> ref(M * N, 0.f);
    for (auto& v : hA) v = __float2bfloat16((float)dist(rng));
    for (auto& v : hB) v = __float2bfloat16((float)dist(rng));
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            for (int k = 0; k < K; k++)
                ref[m * N + n] += __bfloat162float(hA[m * K + k]) *
                                  __bfloat162float(hB[n * K + k]);
    __nv_bfloat16 *dA, *dB;
    float* dD;
    CUDA_CHECK(cudaMalloc(&dA, M * K * 2));
    CUDA_CHECK(cudaMalloc(&dB, N * K * 2));
    CUDA_CHECK(cudaMalloc(&dD, M * N * 4));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), M * K * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), N * K * 2, cudaMemcpyHostToDevice));
    tcgen05_tile<<<1, 128>>>(dA, dB, dD);
    CUDA_CHECK_KERNEL();
    std::vector<float> got(M * N);
    CUDA_CHECK(cudaMemcpy(got.data(), dD, M * N * 4, cudaMemcpyDeviceToHost));
    long bad = 0;
    for (int i = 0; i < M * N; i++)
        if (got[i] != ref[i]) {
            if (bad < 5)
                printf("MISMATCH D[%d][%d]: got %.1f want %.1f\n", i / N,
                       i % N, got[i], ref[i]);
            bad++;
        }
    printf(bad ? "FAIL seed=%u: %ld / %d\n" : "PASS seed=%u\n", seed,
           bad ? bad : (long)seed, M * N);
    return bad != 0;
}