#pragma once
#include <cooperative_groups.h>

// BM rows per CTA; group 2 splits the B tile along N across the pair.
constexpr int BROWS = BN / CTA_GROUP;
constexpr int STAGE_BYTES = (BM + BROWS) * BK * 2;
static_assert(NSTAGE >= 1, "STAGES must be positive");

__device__ inline void issue_copy(int it, int tileM, int tileN, int rank,
                                  uint32_t stage, uint32_t full,
                                  const CUtensorMap* a, const CUtensorMap* b) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :: "r"(full), "r"(STAGE_BYTES) : "memory");
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3}], [%4];"
                 :: "r"(stage), "l"(a), "r"(it*BK), "r"(tileM), "r"(full) : "memory");
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3}], [%4];"
                 :: "r"(stage+BM*BK*2), "l"(b), "r"(it*BK),
                    "r"(tileN + rank*BROWS), "r"(full) : "memory");
}

__global__ void gemm_pipeline(const __nv_bfloat16* gA, const __nv_bfloat16* gB,
                              float* gD, int M, int N, int K,
                              const __grid_constant__ CUtensorMap tmapA,
                              const __grid_constant__ CUtensorMap tmapB) {
    extern __shared__ __align__(1024) uint8_t smem[];
    __shared__ __align__(8) uint64_t full[NSTAGE], empty[NSTAGE];
    __shared__ uint32_t taddr;
    const int tid=threadIdx.x, warp=tid/32, lane=tid%32;
    const int rank = CTA_GROUP == 2 ? blockIdx.x % 2 : 0;
    const int tileM = blockIdx.x*BM, tileN=blockIdx.y*BN;
    const int iters=K/BK;
    uint32_t fa[NSTAGE], ea[NSTAGE];
    const uint32_t base=__cvta_generic_to_shared(smem);
    for (int s=0;s<NSTAGE;++s) {
        fa[s]=__cvta_generic_to_shared(&full[s]);
        ea[s]=__cvta_generic_to_shared(&empty[s]);
        if (tid==0) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(fa[s]) : "memory");
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(ea[s]) : "memory");
        }
    }
    if(tid==0) asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    if(warp==0) {
        uint32_t dst=__cvta_generic_to_shared(&taddr);
#if CTA_GROUP == 1
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(dst),"r"(BN) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
#else
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(dst),"r"(BN) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
#endif
    }
#if CTA_GROUP == 2
    cooperative_groups::this_cluster().sync();
#else
    __syncthreads();
#endif
    int next=0;
    if(tid==0) {
        for(;next<iters && next<NSTAGE;++next)
            issue_copy(next,tileM,tileN,rank,base+next*STAGE_BYTES,fa[next],&tmapA,&tmapB);
    }
    constexpr uint32_t idesc=(1u<<4)|(1u<<7)|(1u<<10)|((BN>>3)<<17)|((BM*CTA_GROUP>>4)<<24);
#if CTA_GROUP == 1
    if(tid==0) {
#endif
    for(int it=0;it<iters;++it) {
        const int s=it%NSTAGE;
        if(tid==0) {
            // Mandatory current tile: the consumer can never wait for an unissued copy.
            if(next==it) {
                if(next>=NSTAGE) mbar_wait(ea[s],((next/NSTAGE)-1)&1);
                issue_copy(next,tileM,tileN,rank,base+s*STAGE_BYTES,fa[s],&tmapA,&tmapB);
                ++next;
            }
            // Only speculative deeper prefetch is nonblocking.
            while(next<iters && next<it+NSTAGE) {
                const int p=next%NSTAGE;
                if(next>=NSTAGE && !mbar_try(ea[p],((next/NSTAGE)-1)&1)) break;
                issue_copy(next,tileM,tileN,rank,base+p*STAGE_BYTES,fa[p],&tmapA,&tmapB);
                ++next;
            }
            mbar_wait(fa[s],(it/NSTAGE)&1);
        }
#if CTA_GROUP == 2
        // Publish both CTAs' TMA completion before the leader reads their DSMEM.
        cooperative_groups::this_cluster().sync();
#endif
        if(tid==0 && rank==0) {
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            for(int kk=0;kk<BK;kk+=16) {
                uint64_t da=make_desc_sm100(base+s*STAGE_BYTES+kk*2,0,1024,2);
                uint64_t db=make_desc_sm100(base+s*STAGE_BYTES+BM*BK*2+kk*2,0,1024,2);
                uint32_t acc=(it!=0 || kk!=0);
#if CTA_GROUP == 1
                asm volatile("{.reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;}"
#else
                asm volatile("{.reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;}"
#endif
                             :: "r"(taddr),"l"(da),"l"(db),"r"(idesc),"r"(acc) : "memory");
            }
#if CTA_GROUP == 1
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(ea[s]) : "memory");
#else
            asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
                         :: "r"(ea[s]),"h"(uint16_t(3)) : "memory");
#endif
        }
    }
#if CTA_GROUP == 1
    }
    __syncthreads(); // Other warps must not observe an earlier generation of the final barrier.
#else
    cooperative_groups::this_cluster().sync();
#endif
    mbar_wait(ea[(iters-1)%NSTAGE],((iters-1)/NSTAGE)&1);
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    for(int c=0;c<BN;c+=8) {
        float r[8];
        const uint32_t src=taddr+((warp*32)<<16)+c;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                     : "=f"(r[0]),"=f"(r[1]),"=f"(r[2]),"=f"(r[3]),"=f"(r[4]),"=f"(r[5]),"=f"(r[6]),"=f"(r[7]) : "r"(src));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        for(int j=0;j<8;++j) gD[(size_t)(tileM+warp*32+lane)*N+tileN+c+j]=r[j];
    }
#if CTA_GROUP == 1
    __syncthreads();
    if(warp==0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(taddr),"r"(BN) : "memory");
#else
    cooperative_groups::this_cluster().sync();
    if(warp==0) asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(taddr),"r"(BN) : "memory");
#endif
}
