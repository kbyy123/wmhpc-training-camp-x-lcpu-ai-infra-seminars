// 问题 2.5：找 bug。
// 这个程序不报错，直接 FAIL（kernel 好像压根没跑。。。）
// 任务：先定位到具体error（提示在文件末尾），再解释原因，并修好它。
#include "common.h"

__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

int main() {
    const int n = 1000003;
    size_t bytes = (size_t)n * sizeof(float);

    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);
    fill_random(h_a, n, 1);
    fill_random(h_b, n, 2);
    for (int i = 0; i < n; i++) h_ref[i] = h_a[i] + h_b[i];

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_c, 0, bytes));

    // int threads = 2048;
    int threads = 1024;
    int blocks = (n + threads - 1) / threads;
    vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    // CUDA_CHECK_KERNEL();
    // 注意：这里故意没有做任何错误检查。

    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    REPORT(check_close(h_c, h_ref, n));
    return 0;
}

// 提示：kernel 启动语句后面补一行 CUDA_CHECK_KERNEL() 再跑一次，
// 报错信息会告诉你该往哪个方向查。查完记得回答：为什么不加这一行时
// 程序一声不吭？（问题 0.2 打印过的哪个上限和这里有关？）

// - 每个 block 启动了 2048 个线程，超过设备上限 1024，触发 cudaErrorInvalidConfiguration，kernel 根本没有执行。
// - <<<...>>> 不返回错误码，也不会自动打印错误；未调用 cudaGetLastError()，所以程序只表现为结果 FAIL。
// - kernel launch 对 host 异步。cudaGetLastError()只能立即检查启动配置；非法访存等错误要等 GPU 实际执行后才产生，因此需要同步才能可靠捕获。
// - 正确检查方式：

// kernel<<<blocks, threads>>>(...);
// CUDA_CHECK(cudaGetLastError());       // 检查启动配置错误
// CUDA_CHECK(cudaDeviceSynchronize());  // 等待执行，并检查异步执行错误