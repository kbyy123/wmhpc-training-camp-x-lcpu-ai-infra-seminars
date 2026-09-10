from pathlib import Path
import importlib.util
import tilelang

# 加载 assignment01 的 make_matmul，避免 kernels 包重名
root = Path(__file__).resolve().parent
src = root / "cuda/m6_tilelang/tilelang_matmul.py"

spec = importlib.util.spec_from_file_location("a1_matmul", src)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# 构造 TileLang kernel；这里不创建 GPU 张量，也不运行计算
func = module.make_matmul(
    1024, 1024, 1024,
    BLOCK_M=128,
    BLOCK_N=128,
    BLOCK_K=64,
    threads=128,
    num_stages=3,
)

out = root / "lowering_output"
out.mkdir(exist_ok=True)

for arch in ["sm_90a", "sm_100a"]:
    target = tilelang.tvm.target.Target({
        "kind": "cuda",
        "arch": arch,
    })

    with target:
        result = tilelang.lower(
            func,
            target=target,
            enable_device_compile=True,
        )

    (out / f"{arch}.cu").write_text(result.kernel_source)
    (out / f"{arch}.device.tir").write_text(result.device_mod.script())
    (out / f"{arch}.host.tir").write_text(result.host_mod.script())

    print(f"{arch}: 编译完成")