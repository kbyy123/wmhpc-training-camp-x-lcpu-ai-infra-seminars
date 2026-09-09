#pragma once
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include "nvfp4_common.h"

__device__ __forceinline__ void load_group(const __nv_bfloat16* p, float (&v)[16]) {
    const uint4 a=*reinterpret_cast<const uint4*>(p);
    const uint4 b=*reinterpret_cast<const uint4*>(p+8);
    const __nv_bfloat162* ah=reinterpret_cast<const __nv_bfloat162*>(&a);
    const __nv_bfloat162* bh=reinterpret_cast<const __nv_bfloat162*>(&b);
#pragma unroll
    for(int i=0;i<4;++i) {
        const float2 x=__bfloat1622float2(ah[i]), y=__bfloat1622float2(bh[i]);
        v[2*i]=x.x; v[2*i+1]=x.y; v[8+2*i]=y.x; v[9+2*i]=y.y;
    }
}

__device__ __forceinline__ void quant_group(const float (&v)[16], uint8_t* data, uint8_t* sf) {
    float amax=0;
#pragma unroll
    for(int i=0;i<16;++i) amax=fmaxf(amax,fabsf(v[i]));
    const __nv_fp8_e4m3 sf8(amax/6.0f);
    *sf=sf8.__x;
    const float scale=float(sf8), inv=scale!=0 ? 1.0f/scale : 0.0f;
    uint2 packed={0,0};
#pragma unroll
    for(int j=0;j<8;++j) {
        const __nv_fp4x2_e2m1 q(make_float2(v[2*j]*inv,v[2*j+1]*inv));
        if(j<4) packed.x |= uint32_t(q.__x)<<(8*j);
        else packed.y |= uint32_t(q.__x)<<(8*(j-4));
    }
    *reinterpret_cast<uint2*>(data)=packed;
}
