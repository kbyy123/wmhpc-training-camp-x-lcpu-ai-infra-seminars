// Assignment 01 Typst edition
// 默认生成带答题框的版本；对照原 handout 时可传入 --input answers=false。
// 作答方法：把各处 #answer[...] 方括号里的提示文字替换为你的答案。
// 如果答案较长，可把相邻的 height: 22mm 调大；没有 height 参数时默认 14mm。
// 表格题把 #fill-cell() 换成 [你的内容]；代码答案仍应写入题目指定的源码文件。
// Typst Web App 可直接上传本文件；本地编译：typst compile assignment01.typ assignment01-typst.pdf

#let answer-mode = sys.inputs.at("answers", default: "true") == "true"
#let accent = rgb("#1c3c78")
#let code-bg = rgb("#f8f8f6")
#let soft-gray = rgb("#f5f5f5")
#let rule-gray = rgb("#888888")
#let section-state = state("section-title", "0 环境准备")

#set document(title: "作业 1：GPU 与 GPU 编程")
#set page(
  paper: "a4",
  margin: (x: 2.3cm, top: 2.4cm, bottom: 2.15cm),
  header: context {
    set text(font: ("Latin Modern Roman", "Noto Serif CJK SC"), size: 8.5pt)
    let title = section-state.get()
    if counter(page).get().first() > 0 {
      [#title #h(1fr) #counter(page).display()]
    }
  },
  numbering: none,
)
#set text(font: ("Latin Modern Roman", "Noto Serif CJK SC"), size: 11.6pt, lang: "zh")
#set par(justify: true, first-line-indent: 2em, leading: .94em, spacing: 1em)
#set list(indent: 2em, body-indent: .6em, spacing: .36em)
#set enum(numbering: "(a)", indent: 2em, body-indent: .6em, spacing: .36em)
#set heading(numbering: none)
#show emph: set text(style: "italic")
#show raw.where(block: true): it => block(
  width: 100%,
  fill: code-bg,
  stroke: .4pt + rgb("#aaaaaa"),
  inset: (x: 7pt, y: 6.5pt),
  radius: 1pt,
  breakable: true,
  above: .55em,
  below: .65em,
  text(font: ("Latin Modern Mono", "Noto Sans Mono CJK SC"), size: 8.9pt, it),
)
#show raw.where(block: false): it => box(
  fill: rgb("#f3f3f3"),
  inset: (x: 1.5pt, y: .3pt),
  radius: 1pt,
  text(font: ("Latin Modern Mono", "Noto Sans Mono CJK SC"), size: .88em, it),
)

#let section-title(number, title) = {
  section-state.update(if number == none { title } else { number + " " + title })
  v(1.1em)
  align(center, text(font: ("Latin Modern Roman", "Noto Serif CJK SC"), size: 15pt, weight: "bold")[
    #if number != none { number + " " }#title
  ])
  v(.55em)
}

#let subsection-title(title) = {
  v(.75em)
  text(font: ("Latin Modern Roman", "Noto Serif CJK SC"), size: 12pt, weight: "bold", title)
  v(.2em)
}

#let titled-box(title, body, accent-box: false) = block(
  width: 100%,
  fill: if accent-box { accent.lighten(91%) } else { soft-gray },
  stroke: .5pt + if accent-box { accent.lighten(25%) } else { rule-gray },
  radius: 1.5pt,
  inset: (x: 7pt, y: 5pt),
  breakable: true,
  above: .55em,
  below: .7em,
  [#set par(first-line-indent: 0em, spacing: .45em)
   #text(font: ("Latin Modern Sans", "Noto Sans CJK SC"), size: 8.8pt, weight: "bold", title)
   #h(.8em)#body],
)

#let reading(body) = titled-box("参考资料", body)
#let lookback(body) = titled-box("Editor's Note", body)
#let capstone(title, body) = titled-box(title, body, accent-box: true)

#let prob(number, kind, optional: none, path: none) = {
  v(.85em)
  block(breakable: false)[
    #set par(first-line-indent: 0em)
    #text(weight: "bold")[prob #number]
    #h(.45em)
    #text(font: ("Latin Modern Sans", "Noto Sans CJK SC"), size: 8.8pt)[（#kind#if optional != none [ · #optional]）]
    #if path != none { h(1fr); text(font: "Noto Sans Mono CJK SC", size: 7.8pt, fill: rgb("#666666"), path) }
  ]
  v(.25em)
}

#let answer(body, height: 14mm) = if answer-mode {
  block(
    width: 100%,
    height: height,
    fill: accent.lighten(96%),
    stroke: .45pt + accent.lighten(65%),
    radius: 1.5pt,
    inset: 6pt,
    breakable: true,
    above: .45em,
    below: .65em,
    [#set par(first-line-indent: 0em)
     #text(font: ("Latin Modern Sans", "Noto Sans CJK SC"), size: 8.8pt, fill: accent.lighten(35%), body)],
  )
}

#let fill-cell(body: [在此填写]) = if answer-mode {
  text(font: ("Latin Modern Sans", "Noto Sans CJK SC"), size: 8pt, fill: rgb("#999999"), body)
} else { [] }

#let clean-table(columns, header, cells, widths: 100%) = align(center, block(width: widths)[
  #set text(size: 8.7pt)
  #table(
    columns: columns,
    inset: (x: 5pt, y: 4.4pt),
    stroke: none,
    table.hline(stroke: .7pt),
    table.header(..header.map(h => strong(h))),
    table.hline(stroke: .4pt),
    ..cells,
    table.hline(stroke: .7pt),
  )
])

#align(center)[
  #text(font: ("Latin Modern Roman", "Noto Serif CJK SC"), size: 19pt, weight: "bold")[作业 1：GPU 与 GPU 编程]
  #v(6pt)
  #text(font: ("Latin Modern Sans", "Noto Sans CJK SC"), size: 10pt)[Weiming HPC Training Camp $times$ LCPU AI Infra Seminars · Session 1]
]
#v(.6em)
#line(length: 100%, stroke: .55pt)
#v(.7em)

#align(center, text(font: ("Latin Modern Roman", "Noto Serif CJK SC"), size: 15pt, weight: "bold")[Preface])
#v(.3em)

此次作业为 Session 1 的配套练习，内容覆盖 CUDA Programming Guide#footnote([#link("https://docs.nvidia.com/cuda/cuda-programming-guide/")[https://docs.nvidia.com/cuda/cuda-programming-guide/]，2025 年重排的新版。下文引用记作 Guide。]) 第一、第二部分的全部章节。

共有七种题型：CONCEPT 概念题、FILL-IN 填空题、MODIFY 改造题、DEBUG 修 bug 题、EXPERIMENT 实验题、HANDS-ON 动手题、FROM-SCRATCH“从零实现”题。标 Optional 的为选做题。涉及代码编辑的题目会在标题行右侧标出相应代码文件的路径（相对 `assignment01/`）。FROM-SCRATCH 题会配套判测脚本。

代码在仓库 `assignment01/` 目录，环境配置见其中的 `README.md`。

AI Policy：`CLAUDE.md` 或 `AGENTS.md`。核心原则是“AI 可以帮你理解，但不能替你实现”。

#subsection-title("Content")

#clean-table(
  (auto, 1fr, 1.6fr, 1.45fr),
  ([Module], [主题], [对应阅读], [代码位置]),
  (
    [0], [环境准备], [README.md], [`cuda/m0_env/`],
    [1], [为什么要用 GPU], [Guide 1.1、1.2.1–1.2.2], [`cuda/m1_why_gpu/`],
    [2], [第一个 CUDA 程序], [Guide 2.1 全章、2.6], [`cuda/m2_first_kernel/`],
    [3], [SIMT 执行], [Guide 2.3.1–2.3.2、1.2.2.2], [`cuda/m3_simt/`],
    [4], [存储空间], [Guide 2.3.3–2.3.5、2.3.7、1.2.3], [`cuda/m4_memory/`],
    [5], [计时与异步初步], [Guide 2.5 引言、1.2.3.3], [`cuda/m5_async/`],
    [6], [Tile 视角], [Guide 1.2.2.3、2.4、2.2], [（阅读为主）],
    [7], [TileLang 与 Triton], [TileLang 文档、Triton 教程], [`kernels/` + `tests/`],
    [8], [平台与编译], [Guide 1.3、2.1.1、2.7], [（复用 `cuda/m0_env`）],
    [Bonus], [matmul], [—], [`cuda/bonus/`、`kernels/`],
  ),
)

#section-title("0", "环境准备")

先把环境跑通——编译一个 minimal 的 CUDA 程序，确认工具链和卡都能用；再把这块卡的参数查一遍，后面的 Module 会反复用到。环境配置见 `README.md`。

#reading[`assignment01/README.md`；查硬件参数用 Guide 第五部分附录 Compute Capabilities。]

#prob("0.1", "HANDS-ON", path: [cuda/m0_env/01_hello.cu])
编译并运行第一个程序，它启动 4 个 block、每个 block 8 个线程。

```bash
cd assignment01/cuda
make run/m0_env/01_hello
```

看看输出的结果，建议至少运行 5 次，可以留意输出各行顺序的变化。源码可以先不细读，Module 8 会回头看 `nvcc` 对它做了什么。

#answer[实验记录：粘贴或概括多次运行时观察到的输出顺序。]

#prob("0.2", "FILL-IN", path: [cuda/m0_env/02_device_query.cu])
`02_device_query.cu` 的五个空都是 `cudaDeviceProp` 的字段名，对照 CUDA Runtime API 文档补全，然后编译运行。

```bash
cd assignment01/cuda
make run/m0_env/02_device_query
```

把你这块卡的参数填进下表，并对照 Guide 附录 Compute Capabilities 里对应架构的表，核对 warp 大小和每 SM 最大线程数。

#clean-table(
  (1.2fr, 1fr),
  ([项目], [你的卡]),
  (
    [型号 / compute capability], [#fill-cell()],
    [SM 数量], [#fill-cell()],
    [warp 大小], [#fill-cell()],
    [shared memory / block], [#fill-cell()],
    [最大常驻线程 / SM], [#fill-cell()],
    [显存总量], [#fill-cell()],
  ), widths: 70%,
)

#answer[补充记录：字段出处、架构表核对结果或运行环境。]

#lookback[
本模块两道题看上去只是热身，但 prob 0.2 的那张表后面会反复被引用：Module 3 讨论 warp 要用 warp 大小，Module 4 手算驻留 block 数要用“shared memory / SM”与“最大常驻线程 / SM”。Keep it at hand.

另外，我们将会逐渐领悟到：一段在 GPU 上运行的代码，其性能与硬件型号是强绑定的。同一段代码在不同 compute capability 上的性能可以相差数倍，某些优化只在某几代 compute capability 上奏效。此后每记录一个耗时，请将硬件型号与 compute capability 一同记下。
]

#section-title("1", "为什么要用 GPU")

GPU 与 CPU 的根本区别在设计目标——CPU 为单线程延迟服务，GPU 为总吞吐服务。此 Module 关注延迟/吞吐、SIMD/SIMT 等相关概念。

#reading[Guide 1.1、1.2.1–1.2.2；LCPU 讲义#footnote(link("https://github.com/interestingLSY/CUDA-From-Correctness-To-Performance-Code/blob/master/lecture.md")[https://github.com/interestingLSY/CUDA-From-Correctness-To-Performance-Code/blob/master/lecture.md]) Part 0。]

#prob("1.1", "CONCEPT")
判断对错，可以顺带补一句理由。

+ 一块标称 100 TFLOPS 的 GPU，执行单条指令的延迟一定低于 5 GHz 的 CPU。
+ HBM 的“高带宽”指大块连续访问时的吞吐，零散的随机访问达不到标称值。
+ 严格串行的迭代算法（每步依赖上一步的结果），即使换一块算力更强的 GPU 也快不了多少。
+ “算力 1000 TFLOPS”意味着每次运算的延迟是 $10^(-15)$ 秒。

#answer(height: 26mm)[(a) 判断与理由：\
(b) 判断与理由：\
(c) 判断与理由：\
(d) 判断与理由：]

#prob("1.2", "CONCEPT")
Session 1 讲座里提过“N 方过百万”这个例子。总计算量 $10^12$ FLOP 在当代 GPU 上的运算时间大概是毫秒级，那为什么一个严格在线的串行算法仍然做不到几秒内跑完？（从“延迟”和“吞吐”的角度考虑）

#answer(height: 22mm)[在这里作答。]

#prob("1.3", "CONCEPT")
补全下表（thread 一行已填好作为示例）。

#clean-table(
  (1fr, 1.7fr, 1.35fr, 1.35fr, 1.5fr),
  ([执行层次], [软件含义], [对应硬件], [直接可用的存储], [同步与通信手段]),
  (
    [thread], [kernel 的最小执行单位], [计算单元上的一个 lane], [自己的寄存器], [（自身天然有序）],
    [warp], [#fill-cell()], [#fill-cell()], [#fill-cell()], [#fill-cell()],
    [block / CTA], [#fill-cell()], [#fill-cell()], [#fill-cell()], [#fill-cell()],
    [grid], [#fill-cell()], [#fill-cell()], [#fill-cell()], [#fill-cell()],
  ),
)

#prob("1.4", "CONCEPT")
SIMD 与 SIMT 的区别？另：判断正误——Nvidia GPU 在 Volta 之后每个线程有独立的 program counter，所以 branch divergence 不再有性能代价。

#answer(height: 23mm)[在这里作答。]

#prob("1.5", "EXPERIMENT", path: [cuda/m1_why_gpu/01_scaling.cu])
同一个向量加法，四种配置各跑一遍。

```bash
cd assignment01/cuda
make run/m1_why_gpu/01_scaling
```

将数据填入下表，并回答：(a) GPU 单线程为什么比 CPU 慢这么多？(b) 从单 block 到铺满 grid 的提速，说明 GPU 加速计算靠的是什么？

#clean-table(
  (1.6fr, 1fr, 1fr),
  ([配置], [耗时 (ms)], [ns / 元素]),
  (
    [CPU 单线程], [#fill-cell()], [#fill-cell()],
    [GPU `<<<1,1>>>`], [#fill-cell()], [#fill-cell()],
    [GPU `<<<1,256>>>`], [#fill-cell()], [#fill-cell()],
    [GPU 铺满 grid], [#fill-cell()], [#fill-cell()],
  ), widths: 75%,
)

#answer(height: 22mm)[(a) 分析：\
(b) 分析：]

#prob("1.6", "FROM-SCRATCH", optional: "Optional", path: [kernels/simt_sim.py])
SIMT Simulator——写一个 warp 的执行模拟器：32 个 lane、共享控制流、带 mask 的分支执行与汇合。请在 `kernels/simt_sim.py` 内完成（docstring 里面有具体实现要求）。工作量大概是 50 行以内的 Python。不需要 GPU。注：本题为 Module 3 的实验结果提供理论对照。判测命令如下：

```bash
cd assignment01
uv run pytest tests/test_simt_sim.py
```

#answer[完成情况、测试结果与实现思路摘要（代码仍写在指定的 `.py` 文件中）。]

#lookback[
延迟与吞吐的分野根植于设计取舍，而非工艺差异。把“同一笔晶体管预算”投向不同的方向：一边是乱序执行、分支预测与庞大的 cache，另一边则是更多计算单元和更宽的访存带宽。prob 1.5 用四档配置让这个判断变得可度量，prob 1.1 与 prob 1.4 分别从两侧划出了它的边界，选做的 prob 1.6 则预先准备了一份理论标尺，留待 Module 3 的实测去校准。

更值得记住的是 prob 1.2 得出的那条结论：无论可用吞吐有多高，它始终绕不过一条串行依赖链。判断某个任务是否值得搬上 GPU，最先看的不是总计算量，而是依赖“图”的宽度——Amdahl 定律在 GPU 场景下的变体大抵如此。至于 prob 1.5 的那份向量加法，它几乎榨不出这块卡真正的算力，真正的瓶颈藏在另一个地方；Module 4 会为你把它量化出来。
]

#section-title("2", "第一个 CUDA 程序")

此 Module 把一个 CUDA 程序的完整生命周期过一遍，覆盖 Guide 2.1 的每个小节。

#reading[Guide 2.1 全章（主要出处）、2.6（浏览 unified memory 相关小节）；LCPU 讲义 Part 1。]

#prob("2.1", "FILL-IN", path: [cuda/m2_first_kernel/01_vector_add.cu])
请填空：`m2_first_kernel/01_vector_add.cu`，具体要求见代码注释。判测命令如下：

```bash
cd assignment01/cuda
make run/m2_first_kernel/01_vector_add
```

#answer[完成情况、测试输出与必要说明。]

#prob("2.2", "CONCEPT")
为下列五个场景选择正确的修饰符（如 `__global__` 等）。

+ 在 GPU 上执行、由 CPU 侧启动的 kernel 函数。
+ 只会被 kernel 调用的辅助函数。
+ host 和 device 代码都要调用的小工具函数。
+ 整个 kernel 运行期间不变、所有线程都要读的系数表。
+ block 内线程共享的暂存数组。

#answer(height: 27mm)[(a) \
(b) \
(c) \
(d) \
(e)]

#prob("2.3", "MODIFY", path: [cuda/m2_first_kernel/02_vector_add_um.cu])
`02_vector_add_um.cu` 代码完整，但目前是“显式内存管理”的版本。改之前先按原样编译运行一次，记下耗时——这一版会被你的改动覆盖掉，下面 (b) 要拿它做对照。然后按文件头的说明改成 unified memory 版（`cudaMallocManaged`），并保持文件头写明的计时窗口不变。

```bash
cd assignment01/cuda
make run/m2_first_kernel/02_vector_add_um
```

然后请回答如下问题：(a) kernel 启动之后、CPU 读结果之前，为什么必须有一次同步？在原先的版本里这次同步发生在哪个调用里？(b) 对比两版“搬运 + kernel + 读回”的耗时，分析差距的原因（谁快谁慢都有可能，与使用的卡有关）。

#answer(height: 30mm)[原版耗时：　　unified memory 版耗时：\
(a) \
(b)]

#prob("2.4", "CONCEPT")
判断对错，可以顺带补一句理由。

+ `vectorAdd<<<...>>>(...)` 这条语句返回时，kernel 一定已经执行完毕。
+ 同一个 stream 里，`cudaMemcpy`（device 到 host）会等它前面的 kernel 全部完成后才开始拷贝。
+ kernel 内部的非法访存，会在启动语句处同步地报出来。

#answer(height: 23mm)[(a) 判断与理由：\
(b) 判断与理由：\
(c) 判断与理由：]

#prob("2.5", "DEBUG", path: [cuda/m2_first_kernel/03_bug_launch.cu])
修 bug：`03_bug_launch.cu`，详细内容见相关文件。

```bash
cd assignment01/cuda
make run/m2_first_kernel/03_bug_launch
```

#answer[错误现象、定位过程与测试结果。]

#prob("2.6", "FILL-IN", path: [cuda/m2_first_kernel/04_matrix_add.cu])
`04_matrix_add.cu` 用二维的 block 和 grid 处理 $1000 times 700$ 的矩阵，请填空实现矩阵加法。测试命令：

```bash
cd assignment01/cuda
make run/m2_first_kernel/04_matrix_add
```

#answer[完成情况、测试输出与必要说明。]

#prob("2.7", "MODIFY", path: [cuda/m2_first_kernel/05_grid_stride.cu])
`05_grid_stride.cu` 的 launch 被固定成 `<<<64, 256>>>`，线程总数远小于 $n$，当前 FAIL。在 launch 配置不变的前提下，把 kernel 改成 grid-stride loop，让任意 $n$ 都能 PASS。

```bash
cd assignment01/cuda
make run/m2_first_kernel/05_grid_stride
```

然后请回答——这种写法的价值在哪里？launch 只有 16384 个线程时，性能上要付出什么代价？

#answer(height: 22mm)[在这里作答。]

#prob("2.8", "EXPERIMENT", path: [cuda/m2_first_kernel/06_whoami.cu])
运行下面的程序两三次，观察 16 个 block 打印输出的先后。

```bash
cd assignment01/cuda
make run/m2_first_kernel/06_whoami
```

(a) 顺序由谁决定？(b) 程序的正确性可以依赖 block 的执行顺序吗？这条限制和 Guide 1.1 说的 scalable programming model 有什么关系？

#answer(height: 25mm)[实验现象：\
(a) \
(b)]

#capstone([prob 2.9（FROM-SCRATCH）：SAXPY #h(1fr) #text(font: "Noto Sans Mono CJK SC", size: 7.8pt)[cuda/m2_first_kernel/saxpy.cu]])[
在 `m2_first_kernel/` 下写出完整的 CUDA 程序 `saxpy.cu`，实现 $y arrow.l 2.0 dot x + y$（单精度）。要求如下：

- 不允许 include `common.h`。错误检查宏和 `cudaEvent` 计时都要自己写一遍。
- 命令行用法：`./saxpy <n>`，$n$ 是元素个数。输入数据按固定公式生成（都是 float）：`x[i] = ((i % 2048) - 1024) * 0.5f`，`y[i] = (i % 1024) - 512`。
- kernel 算完把 $y$ 拷回 host，用 double 累加所有 `y[i]`，输出一行 `SUM=<总和>`（用 `printf("SUM=%.0f\n", s)` 这样的格式，同一行里可以再带上 $n$ 和 kernel 毫秒数），SUM 结果将用于对拍检验程序正确性，exit code 应为 0。
- $n = 0$ 时输出 `SUM=0`，exit code 为 0（0 个 block 的 kernel launch 是非法的，特判即可）。
]

判测脚本覆盖 $n in {0, 1, 31, 1024, 1025, 2^20, 2^20 + 3}$，命令如下。

```bash
cd assignment01/cuda/m2_first_kernel
./judge_saxpy.sh saxpy.cu
```

注：SAXPY 即 Single-precision A$dot$X Plus Y。

#answer(height: 20mm)[实现摘要、判测结果与性能记录。]

#lookback[
CUDA C++ 在 C++ 之上添加的语法扩展其实相当克制：几种函数与变量修饰符、一个 kernel 启动语法，以及一组内存管理 API。真正需要你花时间去适应的是另外两件事——host 与 device 各自独立推进的时间线（prob 2.4），以及数据此刻落在谁的地址空间里，你得“显式”地负责（prob 2.3）。作为本模块收尾的 prob 2.9 把这些概念连同索引、边界检查与错误处理一起收进一个不到百行的程序，prob 5.1 还会回过头来检验其中的计时部分。

prob 2.7 和 prob 2.8 值得多花一些时间。你为这两道题写下的理由——一条关于启动配置与问题规模的关系，另一条关于 block 之间的执行顺序——在后面的内容里会以不同面貌反复出现。Module 7 介绍的 Triton 与 TileLang 仍然建立在同一条 block 无序执行的假设之上，只不过不再需要你亲手把它写出来。
]

#section-title("3", "SIMT 执行")

SIMT 给每个线程“独立执行”的表象，硬件却按 32 线程一组共享 instruction fetch 和 issue（取指和发射）。此模块三个实验的主题分别为“divergence 的代价”、“`__syncthreads` 的必要性”、“block 之间为什么无法同步”。

#reading[Guide 2.3.1–2.3.2、1.2.2.2。3.3、3.5 要用 shared memory，用法见 Guide 2.3.3.2 与 1.2.3.2（Module 4 会系统地读这一块）；cooperative groups 见 Guide 2.3.6（只需浏览）。]

#prob("3.1", "CONCEPT")
设 `blockDim = (8, 8, 1)`。

+ `threadIdx = (3, 5, 0)` 的线性编号是多少？它在第几个 warp、warp 内第几个 lane？
+ 这个 block 一共占多少个 warp？
+ 若 `blockDim = (33, 1, 1)`，占几个 warp？这样配置浪费在哪里？

#answer(height: 25mm)[(a) \
(b) \
(c)]

#prob("3.2", "EXPERIMENT", path: [cuda/m3_simt/01_divergence.cu])
`m3_simt/01_divergence.cu` 的两个 kernel 每线程计算量相同，分支划分不同——一个按 thread 编号的奇偶分（同一个 warp 里一半一半），一个按 warp 边界对齐分。请先预测一下哪个版本运行会更快一点，大概快几倍，然后运行验证：

```bash
cd assignment01/cuda
make run/m3_simt/01_divergence
```

请解释实测比值，并回答——若两个分支的计算量一大一小，按 thread 编号奇偶分的 kernel 和按 warp 边界对齐分的 kernel 的运行时间分别由什么决定？

#answer(height: 30mm)[预测：\
实测结果：\
解释：]

#prob("3.3", "EXPERIMENT", path: [cuda/m3_simt/02_sync_matters.cu])
`02_sync_matters.cu` 让每个 block 用 shared memory 把自己的 256 个元素倒序。请按文件开头的注释内容进行实验。

```bash
cd assignment01/cuda
make run/m3_simt/02_sync_matters
```

(a) 为什么注释掉 sync 后代码不能正确地运行？(b) (Optional) 注释掉 sync 后，翻转后的数组错的位置比较随机，但是有些位置一直是对的，试解释原因。（tip：算一算 $t$ 与 $255-t$ 有没有可能落在同一个 warp）

#answer(height: 28mm)[实验现象：\
(a) \
(b)]

#prob("3.4", "CONCEPT")
`__syncthreads` 只能同步本 block 内的 threads，那需要全 grid 同步时，标准做法是什么？

#answer(height: 18mm)[在这里作答。]

#capstone([prob 3.5（FROM-SCRATCH）：block 内归约 #h(1fr) #text(font: "Noto Sans Mono CJK SC", size: 7.8pt)[cuda/m3_simt/03_reduce.cu]])[
在 `03_reduce.cu` 里从零实现两个求和归约 kernel（判测与计时的代码已经写好）。两个 kernel 的 contract 见文件头。PASS 后，试解释实测性能差距的原因。

（Optional）基于两点事实——(a) `__shfl_down_sync` 是 warp 内寄存器级别的线程间数据交换指令，自带同步效果且延迟极小；(b) 归约到最后 32 个元素后，活跃线程若都落在同一个 warp 里，就不再需要 `__syncthreads`（可以想想为什么）——据此试写出第三版优化后的 kernel。测试时只会跑前两版，第三版自己在 `main` 里照着加一次 `run_one` 调用即可。
]

```bash
cd assignment01/cuda
make run/m3_simt/03_reduce
```

#answer(height: 24mm)[测试与计时结果：\
性能差距分析：\
Optional 版本记录：]

#lookback[
SIMT 让你以单线程的风格编写并行代码，但它只承诺正确性语义，不负责性能语义。本模块的题目正好分别对应这层抽象漏出来的三个口子：warp 才是真实的调度单位（prob 3.2），block 内的并行需要显式同步才能看到彼此的写入（prob 3.3），而 block 之间压根没有提供同步方法（prob 3.4）。

“For performance reasoning, you always need to fall back to the warp level.”

prob 3.5 抛出的结论更让人不适：两个版本的加法次数完全相同，耗时却能相差超过一倍。算法复杂度在这里不再是性能的充分描述——这大概是 GPU 编程与经典算法课之间最根本的分歧。选做的第三版则指向另一条路径：不再把 warp 当作需要小心绕开的硬件细节，而是直接把它当作可调用的编程接口。warp-level primitive 与 cooperative groups（Guide 2.3.6）正是这个方向的官方解答。
]

#section-title("4", "存储空间")

数据的存储位置和迁移速度极大地影响着 kernel 的速度。此 Module 旨在对“存储如何影响性能”建立起基础认知。

#reading[Guide 2.3.3–2.3.5、2.3.7、1.2.3。]

#prob("4.1", "CONCEPT")
补全下表：

#clean-table(
  (1.1fr, 1.35fr, 1.2fr, 1.05fr, 1.35fr),
  ([空间], [谁可见], [生命周期], [片上 / 片外], [谁管理]),
  (
    [register], [单个线程], [线程], [片上], [编译器],
    [local], [#fill-cell()], [#fill-cell()], [#fill-cell()], [#fill-cell()],
    [shared], [#fill-cell()], [#fill-cell()], [#fill-cell()], [#fill-cell()],
    [global], [#fill-cell()], [#fill-cell()], [#fill-cell()], [#fill-cell()],
    [constant], [#fill-cell()], [#fill-cell()], [#fill-cell()], [#fill-cell()],
    [L1 / L2 cache], [#fill-cell()], [#fill-cell()], [#fill-cell()], [#fill-cell()],
  ),
)

#prob("4.2", "FILL-IN", path: [cuda/m4_memory/01_stencil.cu])
请填空：`m4_memory/01_stencil.cu`，具体要求见代码注释。测试命令：

```bash
cd assignment01/cuda
make run/m4_memory/01_stencil
```

#answer[完成情况、测试结果与必要说明。]

#prob("4.3", "MODIFY", path: [cuda/m4_memory/02_constant_coeff.cu])
`02_constant_coeff.cu`：按文件头的说明进行修改。修改完后测试：

```bash
cd assignment01/cuda
make run/m4_memory/02_constant_coeff
```

结果会包含两个版本的耗时（差距可能极小，甚至测不出来性能收益——想想为什么），回答 constant cache 真正的优势在哪种访问模式。

#answer(height: 22mm)[两版耗时：\
分析：]

#prob("4.4", "CONCEPT")
判断对错，可以顺带补一句理由。

+ local memory 的“local”指作用域私有，它实际上在片外显存里。
+ 对数组用运行期才知道的下标做索引，可能迫使它被放进 local memory。

#answer(height: 19mm)[(a) 判断与理由：\
(b) 判断与理由：]

#prob("4.5", "FILL-IN", path: [cuda/m4_memory/03_histogram.cu])
请填空：`03_histogram.cu`，具体要求见代码注释。测试命令：

```bash
cd assignment01/cuda
make run/m4_memory/03_histogram
```

#answer[完成情况、测试结果与必要说明。]

#prob("4.6", "MODIFY", path: [cuda/m4_memory/04_histogram_priv.cu])
`04_histogram_priv.cu`：请按要求修改代码。改完测试指令：

```bash
cd assignment01/cuda
make run/m4_memory/04_histogram_priv
```

测试结果包含与 naive 实现相比较的耗时、吞吐。试解释提速来自哪里。

#answer(height: 22mm)[实测数据：\
提速分析：]

#prob("4.7", "EXPERIMENT", path: [cuda/m4_memory/05_bandwidth.cu])
运行实验，并根据实验数据填表。

```bash
cd assignment01/cuda
make run/m4_memory/05_bandwidth
```

观察数据变化趋势，并简析趋势的成因。

#clean-table(
  (1.2fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
  ([stride], [1], [2], [4], [8], [16], [32]),
  ([GB/s], [#fill-cell()], [#fill-cell()], [#fill-cell()], [#fill-cell()], [#fill-cell()], [#fill-cell()]),
  widths: 78%,
)

#answer(height: 20mm)[趋势与成因：]

#prob("4.8", "EXPERIMENT", path: [cuda/m4_memory/06_occupancy.cu])
`06_occupancy.cu`，详细内容见相关文件。实验命令如下：

```bash
cd assignment01/cuda
make run/m4_memory/06_occupancy
```

记录实验数据：

#clean-table(
  (2fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
  ([shared memory / block (KB)], [], [], [], [], [], []),
  (
    [理论驻留 block / SM], [#fill-cell()], [#fill-cell()], [#fill-cell()], [#fill-cell()], [#fill-cell()], [#fill-cell()],
    [occupancy], [#fill-cell()], [#fill-cell()], [#fill-cell()], [#fill-cell()], [#fill-cell()], [#fill-cell()],
    [实测带宽 (GB/s)], [#fill-cell()], [#fill-cell()], [#fill-cell()], [#fill-cell()], [#fill-cell()], [#fill-cell()],
  ),
)

请回答：(a) 用程序开头打印的“shared memory / SM”和“最大常驻线程 / SM”，手算其中一个的驻留 block 数和 occupancy，和 API 的结果对照。(b) 带宽为什么随 occupancy 下降？用“延迟隐藏需要足够多的常驻 warp”组织你的解释。(c) 表中带宽随 occupancy 单调下降，但明显不成正比——从 100% 到 75% 带宽掉了多少？从 37.5% 到 12.5% 又掉了多少？试解释这个差别。

#answer(height: 36mm)[(a) 计算与对照：\
(b) \
(c)]

#lookback[
本模块的八道题围绕同一个问题展开：数据放在哪一层，以什么模式去取。prob 4.1 给出一幅静态的存储层次地图，prob 4.2 与 4.3 在同一段 stencil 上切换存储位置，prob 4.5 与 4.6 在同一个直方图统计上做同样的变换，prob 4.7 和 4.8 则分别量化访问模式与可并发的请求数量。

有三点值得单独记下。其一，优化只在真正的瓶颈处生效。prob 4.3 是一个刻意安排的负结果：教科书罗列的手段用在了不构成瓶颈的地方，收益为零。避免这类空转，需要先估算再动手，估算可以依赖 arithmetic intensity 与 roofline 模型，后面的 session 会展开。其二，你在 prob 4.8 (b) 写下的那句解释有一个正式的名称——Little 定律：维持某一吞吐量所需的在途请求数，等于吞吐与延迟的乘积。其三，occupancy 本身是手段，不是目标。它高只说明调度器有足够多的 warp 可以切换，并不自动保证访存效率；反过来，低 occupancy 配合足够的指令级并行，同样能跑满内存带宽。Volkov 的 _Better Performance at Lower Occupancy_（GTC 2010）是这一观点的经典论述。
]

#section-title("5", "计时与异步初步")

在前面的练习里 `cudaDeviceSynchronize` 反复出现以保证计时的正确性。本 Module 主要关注两点——kernel 启动是异步的；以及如何准确计时。

#reading[Guide 2.5（只读引言）、1.2.3.3。]

#prob("5.1", "EXPERIMENT", path: [cuda/m5_async/01_timing_trap.cu])
运行实验：

```bash
cd assignment01/cuda
make run/m5_async/01_timing_trap
```

并回答下列问题：(a) 哪个数值可以当作 kernel 耗时写进报告？(b) 另外两个各具体测的是什么？

#answer(height: 25mm)[实验数值：\
(a) \
(b)]

#prob("5.2", "CONCEPT")
判断对错，可以顺带补一句理由。

+ 同一个 stream 里的操作按提交顺序执行。
+ kernel 启动后，host 代码立刻继续往下执行。
+ unified memory 下，CPU 访问一页正被 GPU 占用的内存，会触发缺页与页迁移。

#answer(height: 24mm)[(a) 判断与理由：\
(b) 判断与理由：\
(c) 判断与理由：]

#lookback[
Module 5 的两道题分量不算重，但在整个认知链条里恰好落在一个关键的位置。CUDA 里的异步并不是某种需要额外开启的特性，它更像是 kernel launch 的默认语义：你把工作提交到某条 stream 上，host 端立刻返回，device 端的执行则在另一条时间线上独立展开。正因如此，“计时”这件事本身变成了一门需要认真对待的事——warmup、同步点插在哪里、选哪种 timer、重复多少次，任何一环处理得马虎，拿到的数字就失去了意义。这次作业里，这些细节已经由写好的测试框架替你包揽了；等到将来你自己搭建 benchmark，就得把这些一一补回来。

再下一步，异步自然会引出重叠：多 stream、数据拷贝与计算的并发、用 CUDA Graph 把整张任务图一次性提交。这些内容会放在后续 session 里展开，这里只需要建立起一个清晰的 mental model：host 只管提交，device 只管执行，两条时间线各自推进。
]

#section-title("6", "Tile 视角")

Guide 从 13.x 起把 tile 编程作为与 SIMT 并列的第二种官方模型写进正文。tile 模型描述的是“一个 block 对一块数据做什么”，而 block 内 threads 的分工由编译器决定。此 Module 只做概念铺垫和对照阅读。

#reading[Guide 1.2.2.3（必读，篇幅不长）、2.4.1–2.4.6（浏览）、2.2（知道 CUDA Python 这条路线存在即可）。]

#prob("6.1", "CONCEPT")
判断对错，可以顺带补一句理由。

+ tile 是显存里的一块可变区域，kernel 通过指针直接改写它。
+ 对 tile 的一次运算（如两个 tile 相加）由编译器映射到 block 内的多个线程上执行。
+ tile 模型与 SIMT 模型互斥，一个 CUDA 程序只能选一种。

#answer(height: 24mm)[(a) 判断与理由：\
(b) 判断与理由：\
(c) 判断与理由：]

#prob("6.2", "CONCEPT")
下面是 Guide 2.4.6 的 cuTile Python 向量加法：

```python
import cuda.tile as ct

@ct.kernel
def vec_add(a, b, c, TILE: ct.Constant[int]):
    a_view = a.tiled_view((TILE,))
    b_view = b.tiled_view((TILE,))
    c_view = c.tiled_view((TILE,))

    bid = ct.bid(0)
    a_tile = a_view.load((bid,))
    b_tile = b_view.load((bid,))
    c_view.store((bid,), a_tile + b_tile)
```

据此补全下表（Triton 一列可以做完 Module 7 再回来填）：

#clean-table(
  (1.15fr, 1.4fr, 1.4fr, 1.4fr),
  ([], [CUDA SIMT], [cuTile], [Triton]),
  (
    [并行单位], [block 里的 thread], [block], [#fill-cell()],
    [编号], [`blockIdx` / `threadIdx`], [#fill-cell()], [#fill-cell()],
    [数据分工], [线程用全局下标来划分数据], [#fill-cell()], [#fill-cell()],
    [边界处理], [`if` 判断], [#fill-cell()], [#fill-cell()],
  ),
)

#prob("6.3", "CONCEPT")
仍看上面这段代码。(a) “每个线程对应哪个/些元素”由谁决定？(b) 列出一些在 CUDA SIMT 版向量加法里一定会出现、这里完全没体现出的概念。

#answer(height: 24mm)[(a) \
(b)]

#lookback[
tile 并非某个 DSL 的发明。NVIDIA 自己把它与 SIMT 并列写进了 Guide 正文，这本身就说明它是一类稳定的编程方式，而不是一次工具选型。

从 SIMT 转向 tile，真正改变的是抽象层的职责划分：线程到数据的映射交给了编译器，tile 的尺寸却仍由用户来定。这条界线不是随意画下的——前者凭形状就可以机械推导，后者必须依赖硬件参数与实测，编译器暂时还无力接手。prob 7.5 的表格会把这一判据推广到四种模型上。

需要明确的是，抽象层的上移并不会抹掉底层的硬件。coalescing、shared memory 的容量、occupancy 这些约束仍然在起作用，只不过不再由你亲手去处理。这也是本作业先用五个模块讲清楚 SIMT，再引入 DSL 的缘故。
]

#section-title("7", "TileLang 与 Triton")

此 Module 用 Triton / TileLang 重写一部分之前用 CUDA SIMT 实现的 kernel，重心在 TileLang。
注：Triton 题（7.1、7.2、7.8）不需要 GPU 也能完成正确性部分，运行方式见 `README.md`；TileLang 题都需要 GPU，在集群上跑。

#reading[TileLang Language Basics#footnote(link("https://tilelang.com/programming_guides/language_basics.html")[https://tilelang.com/programming_guides/language_basics.html])（建议阅读）；Triton 官方教程 01-vector-add#footnote(link("https://triton-lang.org/main/getting-started/tutorials/")[https://triton-lang.org/main/getting-started/tutorials/])；tilelang-puzzles#footnote(link("https://github.com/tile-ai/tilelang-puzzles")[https://github.com/tile-ai/tilelang-puzzles])。]

#prob("7.1", "FILL-IN", path: [kernels/vector_add.py])
请填空：`kernels/vector_add.py`，具体要求见代码注释。测试命令：

```bash
cd assignment01
uv run pytest tests/test_vector_add.py
```

#answer[完成情况、测试输出与必要说明。]

#prob("7.2", "MODIFY", path: [kernels/fused_op.py])
按文件开头注释要求修改：`kernels/fused_op.py`。测试命令：

```bash
cd assignment01
uv run pytest tests/test_fused_op.py
```

改完回答——与 Module 2 里改 CUDA kernel 相比，这次的改动主要集中在 kernel 的什么部分？主体代码为什么一行都不用动？

#answer(height: 22mm)[测试结果：\
回答：]

#prob("7.3", "FILL-IN", path: [kernels/tilelang_scale_add.py])
请填空：`kernels/tilelang_scale_add.py`，具体要求见代码注释。测试命令：

```bash
cd assignment01
uv sync --extra tilelang
uv run pytest tests/test_tilelang.py -k scale_add
```

#answer[完成情况、测试输出与必要说明。]

#prob("7.4", "FILL-IN", path: [kernels/tilelang_copy2d.py])
请填空：`kernels/tilelang_copy2d.py`，具体要求见代码注释。测试命令：

```bash
cd assignment01
uv run pytest tests/test_tilelang.py -k copy2d
```

填完想一想：2.6 的四个空——行号、列号、边界保护、grid 尺寸——哪些在这里还有对应？没有对应的那个去哪了？

#answer(height: 22mm)[测试结果：\
回答：]

#prob("7.5", "CONCEPT")
补全下表，每个空填“用户”或“编译器”（二者都涉及的要写清楚各自的范围）。

#clean-table(
  (1.6fr, 1fr, 1fr, 1fr, 1fr),
  ([谁负责], [CUDA SIMT], [cuTile], [Triton], [TileLang]),
  (
    [线程到数据的映射], [用户], [#fill-cell()], [#fill-cell()], [#fill-cell()],
    [边界处理], [用户], [#fill-cell()], [#fill-cell()], [#fill-cell()],
    [tile / block 尺寸的选择], [用户], [#fill-cell()], [#fill-cell()], [#fill-cell()],
    [block 内同步], [用户], [#fill-cell()], [#fill-cell()], [#fill-cell()],
  ),
)

#prob("7.6", "FILL-IN", path: [kernels/tilelang_matmul.py])
请填空：`kernels/tilelang_matmul.py`，具体要求见代码注释。测试命令：

```bash
cd assignment01
uv run pytest tests/test_tilelang.py -k matmul
```

填完回答——这五个空涉及到了 shared memory、寄存器 tile、流水，而 Triton 版 matmul（Bonus 的 `kernels/matmul_triton.py`）并未被显式指定，为什么？

#answer(height: 22mm)[测试结果：\
回答：]

#capstone([prob 7.7（FROM-SCRATCH）：softmax in TileLang #h(1fr) #text(font: "Noto Sans Mono CJK SC", size: 7.8pt)[kernels/tilelang_softmax.py]])[
在 `kernels/tilelang_softmax.py` 里从零实现行 softmax，具体要求见文件开头的 docstring。
]

```bash
cd assignment01
uv run pytest tests/test_tilelang_softmax.py
```

#answer[实现摘要、判测结果与必要说明。]

#prob("7.8", "FROM-SCRATCH", optional: "Optional", path: [kernels/softmax.py])
用 Triton 重写 prob 7.7，此题可以不用 GPU。写完后可以试试对比一下此题与 7.7：归约、边界处理、按形状编译，二者各自是由谁处理的（用户显式处理或者编译器隐式处理）？

测试命令：

```bash
cd assignment01
uv run pytest tests/test_softmax.py
```

#answer(height: 22mm)[测试结果：\
对比分析：]

#lookback[
本模块把前面写过的 kernel 用 TileLang 和 Triton 各重写了一遍。两种写法的区别，基本都收在 prob 7.5 那张对比表里：线程到数据的映射归谁管，边界归谁处理，tile 尺寸归谁选定，同步又归谁插入。DSL 的价值并不在于代码更短，而在于它把前两项交给编译器，同时把后两项——那些必须靠实测才能定下来的旋钮——继续留给你。

抽象当然有代价。Bonus 部分的一组数字提醒你，表达能力和实现成熟度需要分开评估：同一段 TileLang 换个版本，跑出的结果就可能不一样。更关键的是，一旦编译器接手的那部分出了问题——layout 不合法、精度对不上、边界被悄悄吃掉——你的排查路径最终还是得退回 SIMT 这一层，去看它究竟生成了什么。DSL 是构筑在 CUDA 之上的一层，从来不是它的替代品。
]

#section-title("8", "平台与编译")

#reading[Guide 1.3 全章（必读）、2.1.1、2.7（浏览）。]

#prob("8.1", "CONCEPT")
判断对错，可以顺带补一句理由。

+ PTX 是 GPU 直接执行的机器码。
+ 只嵌入了 `sm_70` SASS 的可执行文件，能在 compute capability 9.0 的卡上运行。
+ 一个 fatbin 可以同时携带多个架构的 SASS 和 PTX。
+ JIT 编译由驱动在运行时完成。

#answer(height: 27mm)[(a) 判断与理由：\
(b) 判断与理由：\
(c) 判断与理由：\
(d) 判断与理由：]

#prob("8.2", "EXPERIMENT", optional: "Optional", path: [cuda/m0_env/01_hello.cu])
两个编译实验，对象是 Module 0 的 `01_hello.cu`。

(a) 生成只含 `sm_90` SASS 的可执行文件并运行（如果你的卡本身就是 compute capability 9.0，先把 `ARCH_HIGH` 调成 100 或更高再 make），记录报错信息。

```bash
cd assignment01/cuda
make sassonly/m0_env/01_hello
./bin/m0_env/01_hello_sassonly
```

(b) 生成只含 `compute_75` PTX 的版本（CUDA 13 起 `compute_70` 已被移除，Makefile 默认取 75）并运行。能正常运行吗？PTX 是在什么时候、由谁编译成这块卡的机器码的？

```bash
cd assignment01/cuda
make ptxonly/m0_env/01_hello
./bin/m0_env/01_hello_ptxonly
```

#answer(height: 28mm)[(a) 报错与分析：\
(b) 运行结果与回答：]

#prob("8.3", "CONCEPT", optional: "Optional")
请简单说明 Runtime API 与 Driver API 各自的定位。`cudaMalloc` 属于哪个？

#answer(height: 20mm)[在这里作答。]

#lookback[
Module 0 里那个“能跑就行”的 hello，现在被拆开重新看过：源码经 nvcc 分成 host 和 device 两路，device 端产出 PTX 与 SASS，一并打包进 fatbin；运行时 driver 再从中挑选合适的版本，或者对 PTX 做 JIT。

这一模块的实用意义在交付环节。“编译通过”和“能在目标卡上跑起来”是两回事，后者是部署中最常撞上的墙。当你把一个 GPU 程序交付给别人，至少需要说清楚：它支持哪些 compute capability，fatbin 里包含了哪些架构的 SASS，有没有 PTX 兜底，以及首次运行的 JIT 开销落在哪里。写 kernel 与发布软件是两种视角，而这里正是二者的交界。
]

#pagebreak()
#section-title(none, "Bonus（EXPERIMENT · Optional）：matmul")

调参并比较不同 matmul 实现的性能差距。

+ Naive CUDA：`cuda/bonus/matmul.cu`，用 `-DBS` 取 8、16、32 各跑一遍，记录实测数据。

  ```bash
  cd assignment01/cuda
  make run/bonus/matmul
  nvcc -O2 -std=c++17 -I. -arch=native -DBS=8 -o bin/bonus/matmul_bs8 bonus/matmul.cu && ./bin/bonus/matmul_bs8
  ```

+ Tiled Triton：`kernels/matmul_triton.py`（多组 tile 配置 + cuBLAS 对照），记录实测数据。

  ```bash
  cd assignment01
  uv run python -c "from kernels.matmul_triton import bench; bench()"
  ```

+ TileLang：`kernels/tilelang_matmul.py`（即 prob 7.6 补全的成品），调 `bench()` 的 `block_M / block_N / block_K / num_stages`，记录实测数据。

  ```bash
  cd assignment01
  uv sync --extra tilelang
  uv run python -c "from kernels.tilelang_matmul import bench; bench()"
  ```

试分析性能差距的原因，特别是：TileLang 的控制粒度比 Triton 更细，为什么在这组测试里反而略慢？

注：A100 实测参考——naive CUDA 约 2.4 TFLOPS，Triton 约 125 TFLOPS，TileLang 约 118 TFLOPS，cuBLAS 约 173 TFLOPS，而 A100 fp16 tensor core 的标称上限是 312 TFLOPS。Triton 与 TileLang 的数字只是各自 `bench()` 里那几组配置中的最优，不代表工具的性能上限；tilelang 本身也还在快速迭代（本仓库固定在 0.1.12），数字会随版本变化。

#answer(height: 40mm)[Naive CUDA（BS=8/16/32）：\
Triton：\
TileLang：\
cuBLAS：\
性能分析：]

#lookback[
回顾这份作业，主线可以看作三次视角的切换：单线程到 warp（Module 1–3）、计算到数据搬运（Module 4–5）、线程到 tile（Module 6–7）。走完这一遍，你写下的 kernel 应当能保证正确，且不至于慢得离谱。但距离“快”字仍然有一段差距，Bonus 里那组数字恰好把它量化了出来——naive 实现和 cuBLAS 之间，隔着 tensor core、asynchronous copy、software pipelining 以及 profiler-driven tuning，这些内容会在后续的 session 中展开。
]
