// 问题 5.3(b):实现 NVFP4 quant kernel。
//
// 输入 bf16 矩阵 [M, K](K 是 16 的倍数),输出:
//   dataOut:e2m1 打包数据,每行 K/2 byte,低 nibble 放偶数下标元素
//   sfOut:  e4m3 SF,swizzled 布局(偏移用 nvfp4_common.h 的
//           sf_swizzled_offset;整个 SF 张量已在调用前清零)
//
// 每组的计算顺序(判测按同一顺序生成真值,逐 byte 严格相等):
//   amax = 组内 16 个值的绝对值最大
//   sf8  = __nv_fp8_e4m3(amax / 6.0f)
//   sf   = float(sf8)
//   inv  = sf != 0 ? 1.0f / sf : 0.0f
//   nibble[i] = encode(v[i] * inv)
//
// 设备侧的 e2m1 转换直接用 cuda_fp4.h 的 __nv_fp4x2_e2m1(float2 的 .x
// 进低 nibble),它在 sm_100 家族上是单条硬件指令;你在 5.3(a) 写的
// 编码器语义与它一致,host 参考用的就是它。
//
// 组织建议:16 元素 = 32 byte,一个线程恰好负责一个组,天然免掉组内
// 线程协作;quant 没有行间依赖,grid 怎么铺完全自由。
#pragma once
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include "nvfp4_common.h"

template <int BLOCK>
__global__ void nvfp4_quant_kernel(const __nv_bfloat16* __restrict__ in,
                                   uint8_t* __restrict__ dataOut,
                                   uint8_t* __restrict__ sfOut, int M, int K) {
    const int group_per_row = K / 16;
    const int total_groups = M * group_per_row;
    // 一个线程负责 16 个元素 32 byte，转化为 e2m1 为 8 byte
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_groups) return;

    int row = tid / group_per_row;
    int col_group = tid % group_per_row;
    int ktiles = nvfp4_num_ktiles(K);

    float amax = 0.0;
    for (int i = 0; i < 16; i++) {
        float v = __bfloat162float(in[16 * tid + i]);
        amax = fmaxf(amax, fabsf(v)); 
    }
    
    // .__x 表示原始 bit
    __nv_fp8_e4m3 sf8 = __nv_fp8_e4m3(amax / 6.0f);
    float sf = float(sf8);
    float inv = sf != 0 ? 1.0f / sf : 0.0f;
    sfOut[sf_swizzled_offset(row, col_group, ktiles)] = sf8.__x;
    
    // 16 * tid ~ 16 * tid + 15
    #pragma unroll
    for (int i = 0; i < 8; i += 1) {
        float2 pair = make_float2(__bfloat162float(in[16 * tid + 2 * i]) * inv, __bfloat162float(in[16 * tid + 2 * i + 1]) * inv);
        __nv_fp4x2_e2m1 packed(pair);
        dataOut[8 * tid + i] = packed.__x;
    }
}

// 判测和 5.4 会按这个签名调用;grid 大小你自己定,写在这里。
inline void launch_nvfp4_quant(const __nv_bfloat16* in, uint8_t* dataOut,
                               uint8_t* sfOut, int M, int K, int sms) {
    // TODO: 选择 grid/block 并启动 nvfp4_quant_kernel。
    int total_groups = M * K / 16;
    constexpr int BLOCK = 256;
    int grid = (total_groups + BLOCK - 1) / BLOCK; 
    nvfp4_quant_kernel<BLOCK><<<grid, BLOCK>>>(in, dataOut, sfOut, M, K);    
}
