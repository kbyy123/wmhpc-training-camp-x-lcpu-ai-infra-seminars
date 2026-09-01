#include <cuda_fp8.h>
#include "../common.h"
#include <cstdlib>
#include <random>

__device__ int a_row_of(int lane, int i) {
    int r = i / 4;
    int gid = lane / 4;
    return gid + 8 * (r & 1);
}

__device__ int a_col_of(int lane, int i) {
    int r = i / 4;
    int j = i % 4;
    int tid = lane % 4;
    return 4 * tid + 16 * (r >> 1) + j;
}

__device__ int b_row_of(int lane, int i) {
    int r = i / 4;
    int j = i % 4;
    return 4 * (lane % 4) + 16 * r + j;
}

__device__ int b_col_of(int lane, int i) {
    return lane / 4;
}

__device__ unsigned pack4_A(const uint8_t* base, int row, int col) {
    const uint8_t* raw = reinterpret_cast<const uint8_t*>(base + row * 32 + col);
    return unsigned(raw[0]) | (unsigned(raw[1]) << 8) | (unsigned(raw[2]) << 16) | (unsigned(raw[3]) << 24);
}

__device__ unsigned pack4_B(const uint8_t* base, int row, int col) {
    const uint8_t* raw = reinterpret_cast<const uint8_t*>(base + row * 8 + col);
    return unsigned(raw[0]) | (unsigned(raw[8]) << 8) | (unsigned(raw[16]) << 16) | (unsigned(raw[24]) << 24);
}

__global__ void mma_kernel(const uint8_t* A, const uint8_t* B, float* D) {
    int lane = threadIdx.x;
    int group = lane >> 2;      // 行方向的 8 个组
    int tig = lane & 3;         // 组内 4 个线程

    // A fragment:每线程 16 个 fp8, 4 个 b32 寄存器。
    // 寄存器 r 的两个元素:(row, col) 见下标
    unsigned a[4];
    for (int i = 0; i < 4; i++) {
        a[i] = pack4_A(A, a_row_of(lane, 4 * i), a_col_of(lane, 4 * i));
    }

    // B fragment(col 布局,B 在内存里按 [k][n] 行主序存):
    unsigned b[2];
    for (int i = 0; i < 2; i++) {
        b[i] = pack4_B(B, b_row_of(lane, 4 * i), b_col_of(lane, 4 * i));
    }

    float c[4] = {0.f, 0.f, 0.f, 0.f}, d[4];
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));

    D[(group) * 8 + tig * 2] = d[0];
    D[(group) * 8 + tig * 2 + 1] = d[1];
    D[(group + 8) * 8 + tig * 2] = d[2];
    D[(group + 8) * 8 + tig * 2 + 1] = d[3];
}

static int run_path(unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 15);
    uint8_t hA[16 * 32], hB[32 * 8];
    float fA[16 * 32], fB[32 * 8], ref[16 * 8] = {};
    for (int i = 0; i < 16 * 32; i++) {
        __nv_fp8_e4m3 v = __nv_fp8_e4m3((float)(dist(rng) - 8));
        hA[i] = *(uint8_t*)&v;
        fA[i] = float(v);
    }
    for (int i = 0; i < 32 * 8; i++) {
        __nv_fp8_e4m3 v = __nv_fp8_e4m3((float)(dist(rng) - 8));
        hB[i] = *(uint8_t*)&v;
        fB[i] = float(v);
    }
    for (int r = 0; r < 16; r++)
        for (int n = 0; n < 8; n++)
            for (int k = 0; k < 32; k++)
                ref[r * 8 + n] += fA[r * 32 + k] * fB[k * 8 + n];
    uint8_t *dA, *dB;
    float* dD;
    CUDA_CHECK(cudaMalloc(&dA, sizeof(hA)));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(hB)));
    CUDA_CHECK(cudaMalloc(&dD, 16 * 8 * 4));
    CUDA_CHECK(cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice));
    mma_kernel<<<1, 32>>>(dA, dB, dD);
    CUDA_CHECK_KERNEL();
    float got[16 * 8];
    CUDA_CHECK(cudaMemcpy(got, dD, sizeof(got), cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < 16 * 8; i++) bad += got[i] != ref[i];
    cudaFree(dA); cudaFree(dB); cudaFree(dD);
    return bad;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s SEED\n", argv[0]);
        return 2;
    }
    unsigned seed =
          static_cast<unsigned>(std::strtoul(argv[1], nullptr, 10));
    
    int bad = run_path(seed);
    if (bad) {
        printf("MISMATCH: %d\n", bad);
    }
    else {
        printf("PASS\n");
    }
    return 0;
}
