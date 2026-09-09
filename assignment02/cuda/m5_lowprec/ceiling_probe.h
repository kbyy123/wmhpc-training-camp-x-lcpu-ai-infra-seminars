#pragma once
#include <cuda_bf16.h>
#include "nvfp4_common.h"

template<int BLOCK>
__global__ void probe_kernel(const __nv_bfloat16* __restrict__ in,
                             uint8_t* __restrict__ dataOut,
                             uint8_t* __restrict__ sfOut,int M,int K) {
    const int groups=K/16, nt=nvfp4_num_ktiles(K);
    for(size_t g=(size_t)blockIdx.x*BLOCK+threadIdx.x; g<(size_t)M*groups; g+=(size_t)gridDim.x*BLOCK) {
        const uint4 a=*reinterpret_cast<const uint4*>(in+g*16);
        const uint4 b=*reinterpret_cast<const uint4*>(in+g*16+8);
        const uint2 out=make_uint2(a.x^a.z^b.x^b.z,a.y^a.w^b.y^b.w);
        *reinterpret_cast<uint2*>(dataOut+g*8)=out;
        sfOut[sf_swizzled_offset(g/groups,g%groups,nt)]=uint8_t(out.x^out.y);
    }
}
inline void launch_probe(const __nv_bfloat16* in,uint8_t* data,uint8_t* sf,int M,int K,int sms) {
    constexpr int block=128;
    const int count=(M*(K/16)+block-1)/block;
    const int grid=count<sms*8?count:sms*8;
    probe_kernel<block><<<grid,block>>>(in,data,sf,M,K);
}
