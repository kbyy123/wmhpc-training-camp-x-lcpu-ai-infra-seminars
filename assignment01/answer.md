# Assignment 01 作答记录

> 姓名：__________  日期：__________  GPU：__________  Compute Capability：__________
>
> 使用方法：在 `TODO` 处填写答案；代码题的实现写在题目指定的源码文件中，这里只记录思路、测试和实验结果。选做题可按需跳过。

## Module 0：环境准备

### 0.1 HANDS-ON

- [x] `cuda/m0_env/01_hello.cu` 编译、运行成功

### 0.2 FILL-IN

- [x] 已补全 `cuda/m0_env/02_device_query.cu`
- [x] 已编译运行
- [x] 已与 Guide 的 Compute Capabilities 表核对

| 项目 | 我的 GPU | Guide 核对结果/备注 |
| --- | --- | --- |
| 型号 / compute capability | 12.0       | TODO |
| SM 数量 | 26 | TODO |
| warp 大小 | 32         | TODO |
| shared memory / block | 49152 | TODO |
| 最大常驻线程 / SM | 1536 | TODO |
| 显存总量 | 8546484224 | TODO |

## Module 1：为什么要用 GPU

### 1.1 CONCEPT

| 小问 | 判断 | 
| --- | --- | 
| (a) | 错误 | 
| (b) | 正确 | 
| (c) | 正确 | 
| (d) | 错误 | 

### 1.2 CONCEPT

从延迟和吞吐的角度解释严格在线的串行算法为何无法充分利用 GPU：

TODO

### 1.3 CONCEPT

| 执行层次 | 软件含义 | 对应硬件 | 直接可用的存储 | 同步与通信手段 |
| --- | --- | --- | --- | --- |
| thread | kernel 的最小执行单位 | 计算单元上的一个 lane | 自己的寄存器 | 自身天然有序 |
| warp | TODO | TODO | TODO | TODO |
| block / CTA | TODO | TODO | TODO | TODO |
| grid | TODO | TODO | TODO | TODO |

### 1.4 CONCEPT

- SIMD 与 SIMT 的区别：TODO
- “Volta 后每线程有独立 PC，所以 branch divergence 不再有性能代价”：TODO
- 理由：TODO

### 1.5 EXPERIMENT

| 配置 | 耗时 (ms) | ns / 元素 |
| --- | ---: | ---: |
| CPU 单线程 | 4.815 | 1.15 |
| GPU `<<<1,1>>>` | 257.086 | 61.29 |
| GPU `<<<1,256>>>` | 6.464 | 1.54 |
| GPU 铺满 grid | 0.161 | 0.04 |

- (a) GPU 单线程为什么比 CPU 慢：TODO
- (b) 从单 block 到铺满 grid 的提速说明了什么：TODO

### 1.6 FROM-SCRATCH（Optional）

- [ ] 已完成 `kernels/simt_sim.py`
- [ ] `uv run pytest tests/test_simt_sim.py` 通过
- 实现思路/测试备注：TODO

## Module 2：第一个 CUDA 程序

### 2.1 FILL-IN

- [x] 已补全 `cuda/m2_first_kernel/01_vector_add.cu`
- [x] `make run/m2_first_kernel/01_vector_add` 通过

### 2.2 CONCEPT

| 小问 | 修饰符 | 理由/备注 |
| --- | --- | --- |
| (a) GPU 执行、CPU 启动的 kernel | TODO | TODO |
| (b) 只被 kernel 调用的辅助函数 | TODO | TODO |
| (c) host 和 device 都调用的工具函数 | TODO | TODO |
| (d) kernel 期间不变、所有线程读取的系数表 | TODO | TODO |
| (e) block 内线程共享的暂存数组 | TODO | TODO |

### 2.3 MODIFY

- [ ] 已记录修改前的显式内存管理版本耗时
- [ ] 已将代码改为 unified memory 版本
- [ ] 已保持题目要求的计时窗口
- [ ] 修改后测试通过

| 版本 | 搬运 + kernel + 读回耗时 (ms) | GPU / CC | 备注 |
| --- | ---: | --- | --- |
| 显式内存管理 | 64.8ms | TODO | TODO |
| Unified Memory | 18.7ms | TODO | TODO |

- (a) CPU 读取结果前为何必须同步：TODO
- (a) 原版本中的同步发生在哪个调用：TODO
- (b) 两版耗时差距及原因：TODO

### 2.4 CONCEPT

| 小问 | 判断 | 理由 |
| --- | --- | --- |
| (a) kernel launch 返回时 kernel 一定完成 | TODO | TODO |
| (b) 同一 stream 的 D2H `cudaMemcpy` 会等待前序 kernel | TODO | TODO |
| (c) kernel 非法访存会在启动语句处同步报错 | TODO | TODO |

### 2.5 DEBUG

- [x] 已定位 `cuda/m2_first_kernel/03_bug_launch.cu` 中的问题
- [x] 已自行修复
- [x] 测试通过
- 错误现象：TODO
- 原因分析：TODO
- 验证记录：TODO

### 2.6 FILL-IN

- [ ] 已补全 `cuda/m2_first_kernel/04_matrix_add.cu`
- [ ] 已考虑二维索引与边界
- [ ] 测试通过
- 备注：TODO

### 2.7 MODIFY

- [ ] 已在 launch 配置不变时实现 grid-stride loop
- [ ] 对任意题设规模测试通过
- 这种写法的价值：TODO
- 只启动 16384 个线程的性能代价：TODO

### 2.8 EXPERIMENT

- 运行次数：__________
- 观察到的 block 输出顺序：TODO
- (a) 输出顺序由谁决定：TODO
- (b) 正确性能否依赖 block 执行顺序：TODO
- 与 scalable programming model 的关系：TODO

### 2.9 FROM-SCRATCH：SAXPY

- [ ] 已创建 `cuda/m2_first_kernel/saxpy.cu`
- [ ] 未包含 `common.h`
- [ ] 已自行完成错误检查和 `cudaEvent` 计时
- [ ] 已处理 `n = 0`
- [ ] `judge_saxpy.sh` 全部通过

| n | exit code | SUM 对拍 | kernel 耗时 (ms) | 备注 |
| ---: | ---: | --- | ---: | --- |
| 0 | TODO | TODO | — | TODO |
| 1 | TODO | TODO | TODO | TODO |
| 31 | TODO | TODO | TODO | TODO |
| 1024 | TODO | TODO | TODO | TODO |
| 1025 | TODO | TODO | TODO | TODO |
| 2^20 | TODO | TODO | TODO | TODO |
| 2^20 + 3 | TODO | TODO | TODO | TODO |

实现思路/问题记录：TODO

## Module 3：SIMT 执行

### 3.1 CONCEPT

- (a) `threadIdx = (3,5,0)` 的线性编号：TODO
- (a) 所在 warp：TODO；warp 内 lane：TODO
- (b) `blockDim = (8,8,1)` 占用的 warp 数：TODO
- (c) `blockDim = (33,1,1)` 占用的 warp 数：TODO
- (c) 浪费在哪里：TODO

### 3.2 EXPERIMENT

- 预测较快版本：TODO
- 预测倍率：TODO

| 版本 | 耗时 (ms) | 相对倍率 | GPU / CC |
| --- | ---: | ---: | --- |
| 按 thread 奇偶分支 | TODO | TODO | TODO |
| 按 warp 边界分支 | TODO | TODO | TODO |

- 实测比值解释：TODO
- 分支计算量一大一小时，奇偶分版本的时间由什么决定：TODO
- 分支计算量一大一小时，warp 对齐版本的时间由什么决定：TODO

### 3.3 EXPERIMENT

- [ ] 保留同步时测试
- [ ] 按题意注释同步后测试
- 正确版本结果：TODO
- 无同步版本结果/错误现象：TODO
- (a) 无同步时为何不正确：TODO
- (b, Optional) 某些位置一直正确的原因：TODO

### 3.4 CONCEPT

需要全 grid 同步时的标准做法：TODO

### 3.5 FROM-SCRATCH：block 内归约

- [ ] 已实现两个要求的 kernel
- [ ] 正确性测试通过

| 版本 | 耗时 (ms) | 正确性 | GPU / CC | 备注 |
| --- | ---: | --- | --- | --- |
| Kernel 1 | TODO | TODO | TODO | TODO |
| Kernel 2 | TODO | TODO | TODO | TODO |
| 优化版（Optional） | TODO | TODO | TODO | TODO |

- 前两版性能差距原因：TODO
- Optional 优化思路与结果：TODO

## Module 4：存储空间

### 4.1 CONCEPT

| 空间 | 谁可见 | 生命周期 | 片上 / 片外 | 谁管理 |
| --- | --- | --- | --- | --- |
| register | 单个线程 | 线程 | 片上 | 编译器 |
| local | TODO | TODO | TODO | TODO |
| shared | TODO | TODO | TODO | TODO |
| global | TODO | TODO | TODO | TODO |
| constant | TODO | TODO | TODO | TODO |
| L1 / L2 cache | TODO | TODO | TODO | TODO |

### 4.2 FILL-IN

- [ ] 已补全 `cuda/m4_memory/01_stencil.cu`
- [ ] 测试通过
- 备注：TODO

### 4.3 MODIFY

- [ ] 已按文件要求修改 `cuda/m4_memory/02_constant_coeff.cu`
- [ ] 测试通过

| 版本 | 耗时 (ms) | GPU / CC | 备注 |
| --- | ---: | --- | --- |
| 修改前 | TODO | TODO | TODO |
| 修改后 | TODO | TODO | TODO |

- 性能收益为何可能很小或测不出来：TODO
- constant cache 真正有优势的访问模式：TODO

### 4.4 CONCEPT

| 小问 | 判断 | 理由 |
| --- | --- | --- |
| (a) local memory 作用域私有、物理上位于片外 | TODO | TODO |
| (b) 运行期下标访问数组可能使其进入 local memory | TODO | TODO |

### 4.5 FILL-IN

- [ ] 已补全 `cuda/m4_memory/03_histogram.cu`
- [ ] 测试通过
- 备注：TODO

### 4.6 MODIFY

- [x] 已按要求修改 `cuda/m4_memory/04_histogram_priv.cu`
- [x] 测试通过

| 版本 | 耗时 (ms) | 吞吐 | GPU / CC |
| --- | ---: | ---: | --- |
| Naive | 2.5616 | 6.55 GB/s | TODO |
| 修改后 | 0.0471 | 356.08 GB/s | TODO |

提速来源：TODO

### 4.7 EXPERIMENT

实验环境：GPU ________；compute capability ________

| stride | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GB/s | 296.0 | 183.5 | 122.7 | 70.6 | 71.0 | 68.7 |

- 数据趋势：TODO
- 趋势成因：TODO

### 4.8 EXPERIMENT

实验环境：GPU ________；compute capability ________

| shared memory / block (KB) | TODO | TODO | TODO | TODO | TODO | TODO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 理论驻留 block / SM | TODO | TODO | TODO | TODO | TODO | TODO |
| occupancy | TODO | TODO | TODO | TODO | TODO | TODO |
| 实测带宽 (GB/s) | TODO | TODO | TODO | TODO | TODO | TODO |

- (a) 选取的配置：TODO
- (a) 驻留 block 数手算过程：TODO
- (a) occupancy 手算过程：TODO
- (a) 与 API 结果对照：TODO
- (b) occupancy 下降导致带宽下降的原因：TODO
- (c) 100% → 75% 的带宽变化：TODO
- (c) 37.5% → 12.5% 的带宽变化：TODO
- (c) 两段变化不成比例的解释：TODO

## Module 5：计时与异步初步

### 5.1 EXPERIMENT

实验环境：GPU ________；compute capability ________

| 程序输出的计时项 | 数值 | 实际测量的内容 | 能否作为 kernel 耗时 |
| --- | ---: | --- | --- |
| 计时项 1 | TODO | TODO | TODO |
| 计时项 2 | TODO | TODO | TODO |
| 计时项 3 | TODO | TODO | TODO |

- (a) 报告中应采用哪个数值：TODO
- (b) 另外两个分别测量什么：TODO

### 5.2 CONCEPT

| 小问 | 判断 | 理由 |
| --- | --- | --- |
| (a) 同一 stream 操作按提交顺序执行 | TODO | TODO |
| (b) kernel 启动后 host 立即继续执行 | TODO | TODO |
| (c) CPU 访问 GPU 占用的 unified-memory 页面会触发缺页和迁移 | TODO | TODO |

## Module 6：Tile 视角

### 6.1 CONCEPT

| 小问 | 判断 | 理由 |
| --- | --- | --- |
| (a) tile 是显存中由指针直接修改的可变区域 | TODO | TODO |
| (b) tile 运算由编译器映射到 block 内多个线程 | TODO | TODO |
| (c) tile 与 SIMT 模型互斥 | TODO | TODO |

### 6.2 CONCEPT

| 对比项 | CUDA SIMT | cuTile | Triton |
| --- | --- | --- | --- |
| 并行单位 | block 里的 thread | block | TODO |
| 编号 | `blockIdx` / `threadIdx` | TODO | TODO |
| 数据分工 | 线程用全局下标划分数据 | TODO | TODO |
| 边界处理 | `if` 判断 | TODO | TODO |

### 6.3 CONCEPT

- (a) 每个线程对应哪些元素由谁决定：TODO
- (b) CUDA SIMT 版中会出现、cuTile 示例中未体现的概念：
  - TODO
  - TODO
  - TODO

## Module 7：TileLang 与 Triton

### 7.1 FILL-IN

- [ ] 已补全 `kernels/vector_add.py`
- [ ] `uv run pytest tests/test_vector_add.py` 通过
- 备注：TODO

### 7.2 MODIFY

- [ ] 已修改 `kernels/fused_op.py`
- [ ] `uv run pytest tests/test_fused_op.py` 通过
- 相比 Module 2，改动主要集中在 kernel 的哪部分：TODO
- 主体代码为何不需要改动：TODO

### 7.3 FILL-IN

- [ ] 已补全 `kernels/tilelang_scale_add.py`
- [ ] `uv run pytest tests/test_tilelang.py -k scale_add` 通过
- 备注：TODO

### 7.4 FILL-IN

- [ ] 已补全 `kernels/tilelang_copy2d.py`
- [ ] `uv run pytest tests/test_tilelang.py -k copy2d` 通过
- 行号在这里的对应：TODO
- 列号在这里的对应：TODO
- 边界保护在这里的对应：TODO
- grid 尺寸在这里的对应：TODO
- 没有对应的部分去了哪里：TODO

### 7.5 CONCEPT

每格填写“用户”或“编译器”；若二者都涉及，注明各自范围。

| 谁负责 | CUDA SIMT | cuTile | Triton | TileLang |
| --- | --- | --- | --- | --- |
| 线程到数据的映射 | 用户 | TODO | TODO | TODO |
| 边界处理 | 用户 | TODO | TODO | TODO |
| tile / block 尺寸的选择 | 用户 | TODO | TODO | TODO |
| block 内同步 | 用户 | TODO | TODO | TODO |

### 7.6 FILL-IN

- [ ] 已补全 `kernels/tilelang_matmul.py`
- [ ] `uv run pytest tests/test_tilelang.py -k matmul` 通过
- TileLang 中显式指定 shared memory、寄存器 tile、流水，而 Triton 版未显式指定的原因：TODO

### 7.7 FROM-SCRATCH：softmax in TileLang

- [ ] 已实现 `kernels/tilelang_softmax.py`
- [ ] `uv run pytest tests/test_tilelang_softmax.py` 通过
- 实现思路/问题记录：TODO

### 7.8 FROM-SCRATCH（Optional）：softmax in Triton

- [ ] 已实现 `kernels/softmax.py`
- [ ] `uv run pytest tests/test_softmax.py` 通过

| 对比项 | TileLang | Triton |
| --- | --- | --- |
| 归约由谁处理、如何处理 | TODO | TODO |
| 边界处理由谁处理 | TODO | TODO |
| 按形状编译由谁处理 | TODO | TODO |

其他对比：TODO

## Module 8：平台与编译

### 8.1 CONCEPT

| 小问 | 判断 | 理由 |
| --- | --- | --- |
| (a) PTX 是 GPU 直接执行的机器码 | TODO | TODO |
| (b) 仅含 `sm_70` SASS 的程序可在 CC 9.0 GPU 上运行 | TODO | TODO |
| (c) fatbin 可同时携带多个架构的 SASS 和 PTX | TODO | TODO |
| (d) JIT 编译由驱动在运行时完成 | TODO | TODO |

### 8.2 EXPERIMENT（Optional）

#### (a) SASS-only

- 编译目标架构：TODO
- 本机 GPU / CC：TODO
- 是否成功运行：TODO
- 报错信息：

```text
TODO
```

- 原因分析：TODO

#### (b) PTX-only

- PTX compute 架构：TODO
- 是否成功运行：TODO
- PTX 何时被编译为本机机器码：TODO
- 由谁完成编译：TODO
- 首次运行现象/其他记录：TODO

### 8.3 CONCEPT（Optional）

- Runtime API 的定位：TODO
- Driver API 的定位：TODO
- `cudaMalloc` 属于哪一个：TODO

## Bonus：Matmul（Optional）

实验环境：GPU ________；compute capability ________；数据类型 ________；矩阵形状 ________

### Naive CUDA

| BS | 耗时 (ms) | TFLOPS | 正确性 | 备注 |
| ---: | ---: | ---: | --- | --- |
| 8 | TODO | TODO | TODO | TODO |
| 16 | TODO | TODO | TODO | TODO |
| 32 | TODO | TODO | TODO | TODO |

### Triton / cuBLAS

| 实现/配置 | block M | block N | block K | 其他配置 | 耗时 (ms) | TFLOPS |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Triton 1 | TODO | TODO | TODO | TODO | TODO | TODO |
| Triton 2 | TODO | TODO | TODO | TODO | TODO | TODO |
| Triton 3 | TODO | TODO | TODO | TODO | TODO | TODO |
| cuBLAS | — | — | — | TODO | TODO | TODO |

### TileLang

| 配置 | block M | block N | block K | num stages | 耗时 (ms) | TFLOPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TileLang 1 | TODO | TODO | TODO | TODO | TODO | TODO |
| TileLang 2 | TODO | TODO | TODO | TODO | TODO | TODO |
| TileLang 3 | TODO | TODO | TODO | TODO | TODO | TODO |

- 各实现性能差距及原因：TODO
- TileLang 控制粒度更细但本实验可能较慢的原因：TODO
- 调参结论：TODO

## 最终检查

- [ ] 所有必做代码题通过对应测试
- [ ] 所有概念题已填写
- [ ] 所有实验数据均注明 GPU 型号和 compute capability
- [ ] 所有耗时均注明单位和计时方法
- [ ] Optional 题已明确标注完成或跳过
- [ ] 已运行 CUDA 全量判测
- [ ] 已运行 Python 全量判测

