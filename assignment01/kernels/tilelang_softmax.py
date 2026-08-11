"""问题 7.7（压轴）：softmax in TileLang（FROM-SCRATCH）。

contract：
- softmax(x) 接收形状 (M, N) 的 float32 CUDA tensor，返回同形状结果，
  对每一行独立做 softmax；
- kernel 用 TileLang 自己写，一个 block 处理一行（或一小批行）；
- 为了确保数值稳定，要求行内先减最大值，再做 exp 与求和。测试里有一行
  数值巨大的输入，不稳定的实现会得到 inf/nan；
- 行宽 N 任意，可以假设 N <= 4096。TileLang 的 kernel 按形状编译，
  用 make_xxx(M, N) 针对形状生成、在 wrapper 里按形状缓存编译结果
  是常见做法（结构可以参考 7.3、7.4）；
- 归约用 T.reduce_max / T.reduce_sum，逐元素部分用 T.Parallel 加 T.exp；
- fragment 的宽度建议取不小于 N 的 3 的幂（类比 Triton 的
  next_power_of_2），不足的位置补 -inf（T.if_then_else 加 T.infinity），
  否则布局推断可能报 no available layout；
- 通过 pytest tests/test_tilelang_softmax.py 即为完成。

(Optional) 将你的实现和 torch.softmax 比较一下性能（行宽取 256/1024/4096），
Tip: elementwise + 行内归约的 kernel 大概率是带宽瓶颈，可以想想理论上限是多少。
"""

import torch
import tilelang
import tilelang.language as T

@tilelang.jit
def make_softmax(M, N, dtype="float32"):
    @T.prim_func
    def softmax(
        X: T.Tensor((M, N), dtype),
        Y: T.Tensor((M, N), dtype)
    ):
        with T.Kernel(1, M, threads=128) as (bx, by):
            size = 1 << (N - 1).bit_length()
            X_shared = T.alloc_shared((1, size), dtype)

            for i in T.Parallel(size):
                X_shared[0, i] = T.if_then_else(
                    i < N,
                    X[by, i],
                    -T.infinity(dtype),
                )

            max_value = T.alloc_fragment((1,), dtype)
            exp_sum = T.alloc_fragment((1,), dtype)
            T.reduce_max(X_shared, max_value, dim=-1)

            for i in T.Parallel(size):
                X_shared[0, i] -= max_value[0]

            for i in T.Parallel(size):
                X_shared[0, i] = T.exp(X_shared[0, i])

            T.reduce_sum(X_shared, exp_sum, dim=-1)

            for i in T.Parallel(size):
                X_shared[0, i] /= exp_sum[0]

            T.copy(X_shared, Y[by, 0])

    return softmax

def softmax(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    kernel = make_softmax(M, N)
    y = torch.empty_like(x)
    kernel(x, y)
    return y


def bench(M=4096, widths=(256, 1024, 4096), warmup=25, rep=100):
    """Compare end-to-end TileLang and PyTorch softmax latency.

    The first TileLang invocation for each shape performs JIT compilation and is
    intentionally kept outside the timed region. Reported bandwidth counts one
    input read and one output write, so it is an effective bandwidth rather
    than a measurement of every internal memory transaction.
    """
    import triton

    assert torch.cuda.is_available(), "benchmark requires a CUDA GPU"

    torch.manual_seed(0)
    cases = []
    print("正在编译 TileLang kernel 并检查结果……")
    for N in widths:
        x = torch.randn((M, N), device="cuda", dtype=torch.float32)

        # Compile/warm up TileLang once and verify that both implementations
        # produce the same result before measuring their performance.
        got = softmax(x)
        expected = torch.softmax(x, dim=-1)
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)
        torch.cuda.synchronize()
        del got, expected
        cases.append((N, x))

    print("\n测速口径：端到端耗时，包含输出分配，不包含首次 JIT 编译。")
    print("有效带宽仅按一次读取和一次写回估算，数值越高越好。")

    for N, x in cases:
        tilelang_ms = triton.testing.do_bench(
            lambda: softmax(x), warmup=warmup, rep=rep
        )
        torch_ms = triton.testing.do_bench(
            lambda: torch.softmax(x, dim=-1), warmup=warmup, rep=rep
        )

        transferred_bytes = 2 * x.numel() * x.element_size()
        tilelang_gbps = transferred_bytes / (tilelang_ms * 1e-3) / 1e9
        torch_gbps = transferred_bytes / (torch_ms * 1e-3) / 1e9
        relative = (tilelang_ms / torch_ms - 1.0) * 100.0

        if abs(relative) < 3.0:
            direction = "慢" if relative >= 0 else "快"
            conclusion = (
                f"基本持平（TileLang 比 PyTorch {direction} "
                f"{abs(relative):.1f}%）"
            )
        elif relative > 0:
            conclusion = (
                f"PyTorch 更快；TileLang 耗时是 PyTorch 的 "
                f"{tilelang_ms / torch_ms:.2f} 倍（慢 {relative:.1f}%）"
            )
        else:
            conclusion = (
                f"TileLang 更快；PyTorch 耗时是 TileLang 的 "
                f"{torch_ms / tilelang_ms:.2f} 倍"
            )

        print(f"\n形状：M={M}, N={N}, dtype=float32")
        print(f"  TileLang：{tilelang_ms:.4f} ms，{tilelang_gbps:.1f} GB/s")
        print(f"  PyTorch： {torch_ms:.4f} ms，{torch_gbps:.1f} GB/s")
        print(f"  结论：    {conclusion}")


if __name__ == "__main__":
    bench()
