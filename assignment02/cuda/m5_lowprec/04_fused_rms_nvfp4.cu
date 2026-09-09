// 问题 5.4(全作业结束题):融合 rms_norm + NVFP4 quant。
//
// 背景见题面(vLLM issue #25179 / PR #36413):这个融合上游有两个未
// 合入的 PR,卡在"端到端收益在噪声内,没人解释清楚收益去了哪"。
// 你要做的是把 kernel 写出来,并给出那份解释。
//
// 语义:y = rms_norm(x) * w 后直接量化为 NVFP4(不落 bf16 中间值)。
//   rnorm = 1 / sqrt(mean(x_i^2) + eps),后续每组与 5.3(b) 相同
// 字节账:两步(rms 写读中间值)6.56 B/elem,融合 2.56 B/elem,
// 预言加速比 2.56x。你的任务:逐形状实测,并解释实测与预言的差距
// ——每个 M 段的限制因素是什么,证据(ncu 或推算)是什么。
//
// 本文件给出:两步基线(下面的 rms_norm_baseline_kernel + 你 5.3(b)
// 的 quant kernel)、判测、逐形状计时框架。三处要你动手:
//   1. 实现融合 kernel(结构完全自由)
//   2. 基线要公平:baseline 与融合各自把 grid 等配置调到最优再对比
//      (基线吃亏的对比没有意义,上游 PR 的教训之一)
//   3. 报告:逐形状表 + 差距归因
//
// 判测口径:sumsq 归约顺序不同会让极少数处在舍入边界的值翻转,
// 允许 1e-4 比例的 byte 不一致(host 参考的 sumsq 用 double)。
#include <vector>
#include <random>
#include "../common.h"
#include "nvfp4_common.h"
#include "e2m1_encode.h"
#include "nvfp4_quant_kernel.h"
#include "ceiling_probe.h"
#include <limits>

// 给定的两步基线第一步:block-per-row 的 rms_norm,bf16 进出。
// 允许修改或另写(公平基线的一部分:它调多快,对比就有多可信)。
template <int BLOCK>
__global__ void rms_norm_baseline_kernel(const __nv_bfloat16* __restrict__ in,
                                         const __nv_bfloat16* __restrict__ w,
                                         __nv_bfloat16* __restrict__ out,
                                         int M, int K, float eps) {
    __shared__ float red[BLOCK / 32];
    for (int row = blockIdx.x; row < M; row += gridDim.x) {
        const __nv_bfloat16* xr = in + (size_t)row * K;
        float ss = 0.f;
        for (int k = threadIdx.x * 8; k < K; k += BLOCK * 8) {
            float4 raw = *reinterpret_cast<const float4*>(xr + k);
            const __nv_bfloat162* h =
                reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
            for (int i = 0; i < 4; i++) {
                float2 f = __bfloat1622float2(h[i]);
                ss += f.x * f.x + f.y * f.y;
            }
        }
#pragma unroll
        for (int o = 16; o; o >>= 1) ss += __shfl_down_sync(~0u, ss, o);
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
        __syncthreads();
        if (threadIdx.x < 32) {
            ss = threadIdx.x < BLOCK / 32 ? red[threadIdx.x] : 0.f;
#pragma unroll
            for (int o = 16; o; o >>= 1) ss += __shfl_down_sync(~0u, ss, o);
            if (threadIdx.x == 0) red[0] = ss;
        }
        __syncthreads();
        float rnorm = 1.0f / sqrtf(red[0] / K + eps);
        for (int k = threadIdx.x * 8; k < K; k += BLOCK * 8) {
            float4 raw = *reinterpret_cast<const float4*>(xr + k);
            float4 raww = *reinterpret_cast<const float4*>(w + k);
            const __nv_bfloat162* h =
                reinterpret_cast<const __nv_bfloat162*>(&raw);
            const __nv_bfloat162* hw =
                reinterpret_cast<const __nv_bfloat162*>(&raww);
            __nv_bfloat162 o2[4];
#pragma unroll
            for (int i = 0; i < 4; i++) {
                float2 f = __bfloat1622float2(h[i]);
                float2 fw = __bfloat1622float2(hw[i]);
                o2[i] = __floats2bfloat162_rn(f.x * rnorm * fw.x,
                                              f.y * rnorm * fw.y);
            }
            *reinterpret_cast<float4*>(
                const_cast<__nv_bfloat16*>(out) + (size_t)row * K + k) =
                *reinterpret_cast<float4*>(o2);
        }
        __syncthreads();
    }
}


// A row stays in registers through the reduction, normalization and FP4 conversion.
// K<=8192 and BLOCK>=128 imply at most four groups per thread.
template<int BLOCK>
__global__ void fused_rms_kernel(const __nv_bfloat16* __restrict__ in,
                                 const __nv_bfloat16* __restrict__ w,
                                 uint8_t* __restrict__ data,uint8_t* __restrict__ sf,
                                 int M,int K,float eps) {
    __shared__ float red[BLOCK/32];
    constexpr int SLOTS=(8192/16+BLOCK-1)/BLOCK;
    for(int row=blockIdx.x;row<M;row+=gridDim.x) {
        float v[SLOTS][16],ss=0;
#pragma unroll
        for(int j=0;j<SLOTS;++j) {
            const int g=threadIdx.x+j*BLOCK;
            if(g<K/16) {
                load_group(in+(size_t)row*K+g*16,v[j]);
#pragma unroll
                for(int i=0;i<16;++i) ss+=v[j][i]*v[j][i];
            }
        }
        for(int off=16;off;off>>=1) ss+=__shfl_down_sync(~0u,ss,off);
        if((threadIdx.x&31)==0) red[threadIdx.x/32]=ss;
        __syncthreads();
        if(threadIdx.x<32) {
            ss=threadIdx.x<BLOCK/32?red[threadIdx.x]:0;
            for(int off=16;off;off>>=1) ss+=__shfl_down_sync(~0u,ss,off);
            if(threadIdx.x==0) red[0]=ss;
        }
        __syncthreads();
        const float rnorm=1.0f/sqrtf(red[0]/K+eps);
#pragma unroll
        for(int j=0;j<SLOTS;++j) {
            const int g=threadIdx.x+j*BLOCK;
            if(g<K/16) {
                float weight[16];load_group(w+g*16,weight);
#pragma unroll
                for(int i=0;i<16;++i) v[j][i]=v[j][i]*rnorm*weight[i];
                quant_group(v[j],data+(size_t)row*K/2+g*8,
                            sf+sf_swizzled_offset(row,g,nvfp4_num_ktiles(K)));
            }
        }
        __syncthreads();
    }
}

struct Config { int block=128, grid=1; };
static Config fused_cfg, rms_cfg, quant_cfg;
static void launch_fused(const __nv_bfloat16* in,const __nv_bfloat16* w,
                         uint8_t* data,uint8_t* sf,int M,int K,float eps,int sms) {
#define F(B) case B: fused_rms_kernel<B><<<fused_cfg.grid,B>>>(in,w,data,sf,M,K,eps);break
    switch(fused_cfg.block) { F(128);F(256);F(512); }
#undef F
}
static void launch_rms(const __nv_bfloat16* in,const __nv_bfloat16* w,
                       __nv_bfloat16* mid,int M,int K,float eps) {
#define R(B) case B: rms_norm_baseline_kernel<B><<<rms_cfg.grid,B>>>(in,w,mid,M,K,eps);break
    switch(rms_cfg.block) { R(128);R(256);R(512); }
#undef R
}
static void launch_quant(const __nv_bfloat16* mid,uint8_t* data,uint8_t* sf,int M,int K) {
    if(quant_cfg.block==128) nvfp4_quant_kernel<128><<<quant_cfg.grid,128>>>(mid,data,sf,M,K);
    else nvfp4_quant_kernel<256><<<quant_cfg.grid,256>>>(mid,data,sf,M,K);
}
static void launch_two_step(const __nv_bfloat16* in,const __nv_bfloat16* w,
                            __nv_bfloat16* mid,uint8_t* data,uint8_t* sf,
                            int M,int K,float eps,int sms) {
    launch_rms(in,w,mid,M,K,eps); launch_quant(mid,data,sf,M,K);
}
template<class Launch>
static void tune_rows(Config& cfg,int M,int sms,Launch launch,const char* label) {
    float best=std::numeric_limits<float>::infinity();Config winner;
    for(int block:{128,256,512}) for(int factor:{1,2,4,8,0}) {
        cfg={block,factor?std::min(M,sms*factor):M};
        float ms=time_avg_ms(launch,20);
        CUDA_CHECK_KERNEL();
        printf("tune %s M=%d block=%d grid=%d us=%.3f\n",label,M,cfg.block,cfg.grid,ms*1000);
        if(ms<best){best=ms;winner=cfg;}
    }
    cfg=winner;
}

static void host_ref(const std::vector<float>& x, const std::vector<float>& w,
                     int M, int K, float eps, std::vector<uint8_t>& data,
                     std::vector<uint8_t>& sf, bool round_bf16=false) {
    int numKTiles = nvfp4_num_ktiles(K);
    sf.assign(nvfp4_sf_bytes(M,K),0);
    data.assign((size_t)M * K / 2, 0);
    for (int r = 0; r < M; r++) {
        double ss = 0;
        for (int k = 0; k < K; k++) {
            double v = x[(size_t)r * K + k];
            ss += v * v;
        }
        float rnorm = 1.0f / sqrtf((float)(ss / K) + eps);
        for (int g = 0; g < K / NVFP4_GROUP; g++) {
            float vals[NVFP4_GROUP], amax = 0.f;
            for (int i = 0; i < NVFP4_GROUP; i++) {
                int k = g * NVFP4_GROUP + i;
                vals[i] = x[(size_t)r * K + k] * rnorm * w[k];
                if(round_bf16) vals[i]=__bfloat162float(__float2bfloat16(vals[i]));
                amax = fmaxf(amax, fabsf(vals[i]));
            }
            __nv_fp8_e4m3 sf8 = __nv_fp8_e4m3(amax / 6.0f);
            sf[sf_swizzled_offset(r,g,numKTiles)]=sf8.__x;
            float s = float(sf8);
            float inv = s != 0.f ? 1.0f / s : 0.f;
            for (int i = 0; i < NVFP4_GROUP; i += 2)
                data[(size_t)r * K / 2 + g * 8 + i / 2] =
                    (uint8_t)(e2m1_encode(vals[i + 1] * inv) << 4 |
                              e2m1_encode(vals[i] * inv));
        }
    }
}

int main(int argc,char** argv) {
    int onlyM=argc>2?atoi(argv[1]):0, onlyK=argc>2?atoi(argv[2]):0;
    int sms;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    const float eps = 1e-6f;
    printf("# %-6s %-6s %10s %10s %8s\n", "M", "K", "2step_us", "fused_us",
           "speedup");
    long total_bad = 0;
    for (const auto& shape :
         {std::pair{1, 4096}, {16, 4096}, {256, 4096}, {1024, 4096},
          {4096, 4096}, {16384, 4096}, {4096, 7168}, {16384, 7168},
          {4096, 8192}, {16384, 8192}}) {
        int M = shape.first;
        int K = shape.second;
        if(onlyM && (M!=onlyM || K!=onlyK)) continue;
        size_t n = (size_t)M * K;
        int64_t sfB = nvfp4_sf_bytes(M, K);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-2.f, 2.f);
        std::vector<__nv_bfloat16> hx(n), hw(K);
        std::vector<float> hxf(n), hwf(K);
        for (size_t i = 0; i < n; i++) {
            hx[i] = __float2bfloat16(dist(rng));
            hxf[i] = __bfloat162float(hx[i]);
        }
        for (int i = 0; i < K; i++) {
            hw[i] = __float2bfloat16(dist(rng) * 0.5f);
            hwf[i] = __bfloat162float(hw[i]);
        }
        __nv_bfloat16 *dx, *dw, *dmid;
        uint8_t *dd, *dsf;
        CUDA_CHECK(cudaMalloc(&dx, n * 2));
        CUDA_CHECK(cudaMalloc(&dw, (size_t)K * 2));
        CUDA_CHECK(cudaMalloc(&dmid, n * 2));
        CUDA_CHECK(cudaMalloc(&dd, n / 2));
        CUDA_CHECK(cudaMalloc(&dsf, sfB));
        CUDA_CHECK(cudaMemcpy(dx, hx.data(), n * 2, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dw, hw.data(), (size_t)K * 2,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(dsf, 0, sfB));


        tune_rows(fused_cfg,M,sms,[&]{launch_fused(dx,dw,dd,dsf,M,K,eps,sms);},"fused");
        tune_rows(rms_cfg,M,sms,[&]{launch_rms(dx,dw,dmid,M,K,eps);},"rms");
        float bestq=std::numeric_limits<float>::infinity();Config winq;
        for(int block:{128,256}) for(int factor:{4,8,16,0}) {
            int count=(M*(K/16)+block-1)/block;
            quant_cfg={block,factor?std::min(count,sms*factor):count};
            float qt=time_avg_ms([&]{launch_quant(dmid,dd,dsf,M,K);},20);
            CUDA_CHECK_KERNEL();
            printf("tune quant M=%d K=%d block=%d grid=%d us=%.3f\n",M,K,block,quant_cfg.grid,qt*1000);
            if(qt<bestq){bestq=qt;winq=quant_cfg;}
        }
        quant_cfg=winq;
        printf("selected M=%d K=%d fused=%d/%d rms=%d/%d quant=%d/%d\n",
               M,K,fused_cfg.block,fused_cfg.grid,rms_cfg.block,rms_cfg.grid,quant_cfg.block,quant_cfg.grid);
        CUDA_CHECK(cudaMemset(dsf,0,sfB));
        launch_fused(dx, dw, dd, dsf, M, K, eps, sms);
        CUDA_CHECK_KERNEL();
        std::vector<uint8_t> gd(n / 2), rd, gsf(sfB), rsf;
        CUDA_CHECK(cudaMemcpy(gd.data(), dd, n / 2, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gsf.data(),dsf,sfB,cudaMemcpyDeviceToHost));
        host_ref(hxf, hwf, M, K, eps, rd, rsf);
        long bad = 0;
        for (size_t i = 0; i < gd.size(); i++) bad += gd[i] != rd[i];
        long badsf=0;
        for(size_t i=0;i<gsf.size();++i) badsf+=gsf[i]!=rsf[i];
        bool pass = bad <= (long)(gd.size() / 10000) + 1 && badsf <= (long)(M*(K/16)/10000)+1;
        CUDA_CHECK(cudaMemset(dsf,0,sfB));
        launch_two_step(dx,dw,dmid,dd,dsf,M,K,eps,sms);CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaMemcpy(gd.data(),dd,n/2,cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gsf.data(),dsf,sfB,cudaMemcpyDeviceToHost));
        host_ref(hxf,hwf,M,K,eps,rd,rsf,true);
        long badbase=0,badsfbase=0;
        for(size_t i=0;i<gd.size();++i) badbase+=gd[i]!=rd[i];
        for(size_t i=0;i<gsf.size();++i) badsfbase+=gsf[i]!=rsf[i];
        pass &= badbase<=(long)(gd.size()/10000)+1 && badsfbase<=(long)(M*(K/16)/10000)+1;
        total_bad += !pass;

        int iters = M >= 4096 ? 40 : 200;
        float t2 = time_avg_ms(
            [&] { launch_two_step(dx, dw, dmid, dd, dsf, M, K, eps, sms); },
            iters);
        float tf = time_avg_ms(
            [&] { launch_fused(dx, dw, dd, dsf, M, K, eps, sms); }, iters);
        float tp=time_avg_ms([&]{launch_probe(dx,dd,dsf,M,K,sms);},iters);
        printf("details M=%d K=%d fused_sf_bad=%ld baseline_data_bad=%ld baseline_sf_bad=%ld probe_us=%.3f fused_GBs=%.1f probe_GBs=%.1f fused/probe=%.4f\n",
               M,K,badsf,badbase,badsfbase,tp*1000,effective_gbps(n*2.5625,tf),effective_gbps(n*2.5625,tp),tp/tf);
        printf("  %-6d %-6d %10.2f %10.2f %7.2fx %s(bad=%ld)\n", M, K,
               t2 * 1e3, tf * 1e3, t2 / tf, pass ? "PASS" : "FAIL", bad);
        cudaFree(dx); cudaFree(dw); cudaFree(dmid); cudaFree(dd);
        cudaFree(dsf);
    }
    return total_bad != 0;
}
