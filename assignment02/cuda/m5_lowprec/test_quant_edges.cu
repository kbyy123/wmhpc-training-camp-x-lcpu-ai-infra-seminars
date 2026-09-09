// Additional checks for one/tail groups, signed zero, underflow, and the XOR probe.
#include "../common.h"
#include "nvfp4_quant_kernel.h"
#include "e2m1_encode.h"
#include "ceiling_probe.h"
#include <vector>
#include <cstring>
int main() {
    int sms;CUDA_CHECK(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,0));
    long total=0;
    for(auto shape:{std::pair{1,16},std::pair{3,48},std::pair{129,80}}) {
        int M=shape.first,K=shape.second;size_t n=(size_t)M*K;
        std::vector<__nv_bfloat16> in(n);
        for(size_t i=0;i<n;++i) {
            float v=(int(i%17)-8)*0.375f;
            if(i/16%3==0)v=0;
            if(i/16%3==1)v*=0.0001f;
            if(i&1)v=-v;
            in[i]=__float2bfloat16(v);
        }
        __nv_bfloat16* x;uint8_t *d,*sf;
        CUDA_CHECK(cudaMalloc(&x,n*2));CUDA_CHECK(cudaMalloc(&d,n/2));
        CUDA_CHECK(cudaMalloc(&sf,nvfp4_sf_bytes(M,K)));
        CUDA_CHECK(cudaMemcpy(x,in.data(),n*2,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(sf,0,nvfp4_sf_bytes(M,K)));
        std::vector<uint8_t> gd(n/2),gs(nvfp4_sf_bytes(M,K)),rd(n/2),rs(gs.size(),0);
        launch_nvfp4_quant(x,d,sf,M,K,sms);CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaMemcpy(gd.data(),d,gd.size(),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gs.data(),sf,gs.size(),cudaMemcpyDeviceToHost));
        for(int r=0;r<M;++r)for(int g=0;g<K/16;++g) {
            float v[16],am=0;
            for(int i=0;i<16;++i){v[i]=__bfloat162float(in[(size_t)r*K+g*16+i]);am=fmaxf(am,fabsf(v[i]));}
            __nv_fp8_e4m3 s8(am/6.0f);rs[sf_swizzled_offset(r,g,nvfp4_num_ktiles(K))]=s8.__x;
            float inv=float(s8)!=0?1.0f/float(s8):0;
            for(int i=0;i<8;++i)rd[(size_t)r*K/2+g*8+i]=e2m1_encode(v[2*i]*inv)|(e2m1_encode(v[2*i+1]*inv)<<4);
        }
        long bad=0;for(size_t i=0;i<gd.size();++i)bad+=gd[i]!=rd[i];
        for(size_t i=0;i<gs.size();++i)bad+=gs[i]!=rs[i];
        printf("edge quant M=%d K=%d %s bad=%ld\n",M,K,bad?"FAIL":"PASS",bad);total+=bad;
        CUDA_CHECK(cudaMemset(sf,0,gs.size()));
        launch_probe(x,d,sf,M,K,sms);CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaMemcpy(gd.data(),d,gd.size(),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gs.data(),sf,gs.size(),cudaMemcpyDeviceToHost));
        std::fill(rs.begin(),rs.end(),0);
        for(int r=0;r<M;++r)for(int g=0;g<K/16;++g) {
            uint32_t raw[8];memcpy(raw,in.data()+(size_t)r*K+g*16,32);
            uint32_t out[2]={raw[0]^raw[2]^raw[4]^raw[6],raw[1]^raw[3]^raw[5]^raw[7]};
            memcpy(rd.data()+(size_t)r*K/2+g*8,out,8);
            rs[sf_swizzled_offset(r,g,nvfp4_num_ktiles(K))]=uint8_t(out[0]^out[1]);
        }
        bad=0;for(size_t i=0;i<gd.size();++i)bad+=gd[i]!=rd[i];
        for(size_t i=0;i<gs.size();++i)bad+=gs[i]!=rs[i];
        printf("edge probe M=%d K=%d %s bad=%ld\n",M,K,bad?"FAIL":"PASS",bad);total+=bad;
        cudaFree(x);cudaFree(d);cudaFree(sf);
    }
    return total!=0;
}
