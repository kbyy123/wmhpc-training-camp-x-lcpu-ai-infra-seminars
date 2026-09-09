"""问题 5.1:per-tensor scale 与 outlier。

构造一个张量:一万个元素均匀分布在 [-1, 1],外加一个 3000 的
outlier。按 per-tensor 方式量化到 E4M3(scale = amax / 448,cast 用
torch.float8_e4m3fn),反量化后测逐点相对误差,填题面的表并回答三问。

需要动手的是下面两个 TODO;跑法:
    uv run python kernels/quant_outlier.py
输出直接用于报告,没有自动判测。
"""

import torch

E4M3_MAX = 448.0


def build_tensor(n: int = 10000, outlier: float = 3000.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    x = torch.rand(n, generator=g) * 2 - 1
    return torch.cat([x, torch.tensor([outlier])])


def quant_dequant_per_tensor(x: torch.Tensor) -> torch.Tensor:
    """per-tensor E4M3 量化再反量化。

    TODO: 实现。步骤:算 scale = amax / 448;除 scale 后 cast 到
    torch.float8_e4m3fn;cast 回 float 再乘 scale。
    """
    scale = x.abs().amax() / E4M3_MAX
    if scale == 0:
        return x.float().clone()
    return (x / scale).to(torch.float8_e4m3fn).float() * scale


def rel_err_at(x: torch.Tensor, y: torch.Tensor, value: float) -> float:
    """取 x 中最接近 value 的元素,返回该点的相对误差。

    TODO: 实现(表格的每一格都从这里来)。
    """
    idx = (x - value).abs().argmin()
    err = (y[idx] - x[idx]).abs()
    return (err / x[idx].abs()).item() if x[idx] != 0 else err.item()


def main() -> None:
    x = build_tensor()
    y = quant_dequant_per_tensor(x)
    print("含 outlier:")
    for v in (0.5, 0.1, 0.01, 0.005, 3000.0):
        print(f"  x≈{v:<8} rel_err={rel_err_at(x, y, v):.3e}")
    clean = x[:-1]
    yc = quant_dequant_per_tensor(clean)
    e1, e0 = rel_err_at(x, y, 0.5), rel_err_at(clean, yc, 0.5)
    print(f"without_outlier err_at_0.5={e0:.9g} ratio_with/without={e1/e0:.9g}")
    scale = x.abs().amax().item() / E4M3_MAX
    threshold = scale * 2**-10
    probes = torch.tensor([threshold * (1-1e-4), threshold, threshold*(1+1e-4)])
    yp = (probes / scale).to(torch.float8_e4m3fn).float() * scale
    print(f"scale={scale:.9g} zero_threshold={threshold:.9g} probes={yp.tolist()}")
    yb = torch.cat([quant_dequant_per_tensor(b) for b in x.split(128)])
    for label, sl in (("clean_blocks", slice(0,9984)), ("outlier_block_clean", slice(9984,10000))):
        re = (yb[sl]-x[sl]).abs()/x[sl].abs().clamp_min(1e-30)
        rt = (y[sl]-x[sl]).abs()/x[sl].abs().clamp_min(1e-30)
        print(f"{label} mean_relative={re.mean():.9g} max_relative={re.max():.9g} zeros={(yb[sl]==0).sum().item()} per_tensor_mean={rt.mean():.9g}")
    print(f"outlier_block_size={len(x)%128} outlier_rel={rel_err_at(x,yb,3000):.9g}")
    assert torch.equal(quant_dequant_per_tensor(torch.zeros(128)), torch.zeros(128))
    assert torch.equal(yb[9984:], y[9984:])
    # (a) 去掉 outlier 重新量化,对比 0.5 处的误差
    # (b) 找出被量化成 0 的阈值,写出它与 scale 的关系式
    # (c) 换 1x128 的 per-block scale,对比含/不含 outlier 的 block
    # 这三问自己补代码,结果写进报告。


if __name__ == "__main__":
    main()
