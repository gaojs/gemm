# GPU GEMM 优化记录

## 1. 优化目标与基线

目标是在保证双精度矩阵乘法 `C = A × B` 正确的前提下，降低计算和数据传输的总耗时，提高端到端 GFLOPS。

CPU 基线是 `cuda/gemm_cpu.cu` 中的 `cpu-reference`，采用普通三重循环实现 GEMM。GPU 基线是 `cuda/gemm_naive.cu` 中的 `cuda-naive`：每个线程计算一个输出元素，沿 K 维逐项累加。`cublas-dgemm` 是 cuBLAS 库实现，用作对照，不是 GPU 基线；本项目没有使用 TensorRT。

## 2. 源码组织与性能分析

### 源码组织

| 文件 | 作用 |
| --- | --- |
| `cuda/common.cuh` | CUDA/cuBLAS 错误检查宏、公共类型和函数声明 |
| `cuda/common.cu` | 矩阵生成、误差检查、结果输出和公共计时流程实现 |
| `cuda/gemm_cpu.cu` | CPU reference GEMM 实现 |
| `cuda/gpu_runner.cu` | 主入口，依次运行 CPU、GPU 各版本和 cuBLAS |
| `cuda/gemm_cublas.cu` | cuBLAS 双精度 GEMM 实现 |
| `cuda/gemm_naive.cu` | GPU naive 基线 kernel |
| `cuda/gemm_naive_ilp4.cu`、`cuda/gemm_naive_1x2.cu` | naive kernel 的优化实现 |
| `cuda/gemm_tiled_32.cu` | 共享内存分块实现 |
| `cuda/gemm_reg_2x2.cu`、`cuda/gemm_reg_4x4.cu`、`cuda/gemm_reg_4x4_64.cu` | 寄存器分块实现 |
| `cuda/run_benchmark.sh` | 自动选择架构、编译并运行基准程序 |
| `cuda/plot_results.py` | 解析已有结果并生成性能曲线 |

Nsight Systems 用于观察 CUDA API、H2D/D2H 数据传输和 GPU kernel 时间线；Nsight Compute 用于分析指定 kernel 的计算、访存和占用率。下文给出可执行命令和结果判读方法。

### Nsight Systems：时间线与数据搬运

先在可访问 GPU 的普通 Terminal 设置驱动库路径并编译带行号信息的 profiling 程序：

```bash
cd /home/hill/gemm
export LD_LIBRARY_PATH=/usr/lib64:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
for source in common gemm_cpu gemm_naive gemm_naive_ilp4 gemm_naive_1x2 \
  gemm_tiled_32 gemm_reg_2x2 gemm_reg_4x4 gemm_reg_4x4_64 \
  gemm_cublas gpu_runner; do
  nvcc -O3 -std=c++14 -arch=sm_89 -lineinfo -dc \
    -c "cuda/${source}.cu" -o "/tmp/${source}.o"
done
nvcc -O3 -std=c++14 -arch=sm_89 -lineinfo \
  /tmp/common.o /tmp/gemm_cpu.o /tmp/gemm_naive.o \
  /tmp/gemm_naive_ilp4.o /tmp/gemm_naive_1x2.o /tmp/gemm_tiled_32.o \
  /tmp/gemm_reg_2x2.o /tmp/gemm_reg_4x4.o /tmp/gemm_reg_4x4_64.o \
  /tmp/gemm_cublas.o /tmp/gpu_runner.o \
  -lcublas -lcudart -o /tmp/gpu_gemm_profile.x
```

A100 将 `sm_89` 改为 `sm_80`。采集 Nsight Systems 报告并汇总：

```bash
nsys profile --trace=cuda,cublas --sample=none --cpuctxsw=none \
  --output=/tmp/gemm_nsys /tmp/gpu_gemm_profile.x 1024
nsys stats --report cudaapisum --report gpukernsum --report gpumemtimesum \
  /tmp/gemm_nsys.nsys-rep
nsys stats --report gputrace /tmp/gemm_nsys.nsys-rep
```

`gputrace` 用于按启动查看 kernel 和 memcpy 活动；`cudaapisum` 是 CPU 侧 API 耗时，`gpukernsum` 是 GPU kernel 汇总，`gpumemtimesum` 汇总设备拷贝。不要混淆 CPU API 时间与 GPU 执行时间；汇总还混合了 CPU 结果和多个 GPU 版本/尺寸。应结合时间线查看 `cuda-naive` 周围 H2D、kernel、D2H 的耗时构成，判断端到端时间主要花在分配/传输还是计算。该 runner 的 GPU 计时包含分配、拷贝、kernel 和释放，但 nsys 的默认 CUDA trace 不一定将 `cudaMalloc/cudaFree` 展示为与这些同步操作同口径的单一区间；端到端性能数字仍以未挂 profiler 的正式基准为准。可用 `nsys-ui /tmp/gemm_nsys.nsys-rep` 打开报告。

### Nsight Compute：单个 kernel 指标

对 N=1024 的 naive kernel 做基础采样。runner 对每个 GPU 版本先预热、再重复 5 次；`--launch-skip 1` 跳过首次匹配（预热），采集随后第一次正式计时的 naive 启动。该程序只运行 N=1024，因此不会混入其他尺寸：

```bash
ncu --set basic --kernel-name 'regex:gemm_naive' --launch-skip 1 --launch-count 1 \
  --export /tmp/gemm_naive_ncu /tmp/gpu_gemm_profile.x 1024
ncu --import /tmp/gemm_naive_ncu.ncu-rep
```

若当前 NCU 不支持 `basic`，先用 `ncu --list-sets` 查看可用集合并改用 `default`。需要深入指标时可改用 `--set full`（采集开销和指标数量会增加）。检查报告中的 kernel 名、grid/block 是否对应 N=1024。`--launch-count 1` 只限定采集一个匹配启动；NCU 的 replay 可能重复执行该 kernel。不要用 profiler 运行时输出的 GFLOPS 作性能比较。

分析时依次检查：

1. `SpeedOfLight` / `ComputeWorkloadAnalysis`：计算管线是否接近饱和，FP64 吞吐是否为限制；
2. `MemoryWorkloadAnalysis`：DRAM、L2 吞吐和缓存命中，判断是否真是带宽受限；
3. `Occupancy`：寄存器、共享内存是否限制活跃 warps；
4. `WarpStateStats`：warp 等待来自依赖、访存还是同步；
5. 对照 SASS/源代码行，确认指标对应的指令与访存行为。

指标提示是待验证线索，不是瓶颈结论。Profiler 会改变时钟、缓存与执行次数；加速比必须来自未挂分析器、同设备同尺寸的重复基准。

## 3. 优化版本

优化 kernel 独立成章，便于后续增删版本；当前版本、源文件和方法如下：

| 输出版本 | 源文件 | 核心方法 | 主要权衡 |
| --- | --- | --- | --- |
| `cuda-naive-ilp4` | `cuda/gemm_naive_ilp4.cu` | 四条独立累加链 | 减少累加依赖；改变求和顺序 |
| `cuda-naive-1x2` | `cuda/gemm_naive_1x2.cu` | 每线程计算相邻两个 C 元素 | 复用 A；减少线程数 |
| `cuda-tiled-32` | `cuda/gemm_tiled_32.cu` | 32×32 共享内存分块 | 复用 A/B；增加同步和共享内存成本 |
| `cuda-reg-2x2` | `cuda/gemm_reg_2x2.cu` | 每线程计算 2×2 输出块 | 增加数据复用和指令并行性；消耗更多寄存器 |
| `cuda-reg-4x4` | `cuda/gemm_reg_4x4.cu` | 每线程计算 4×4 输出块 | 增加数据复用；寄存器压力更高 |
| `cuda-reg-4x4-64` | `cuda/gemm_reg_4x4_64.cu` | 64×64 输出块，K 方向按 8 分块 | 输出块更大；资源和同步开销可能抵消收益 |

改变累加顺序会带来正常的浮点误差差异，应先确认误差在可接受范围内，再比较性能。

## 4. 测试方法

统一入口为 `cuda/gpu_runner.cu`，由 `cuda/run_benchmark.sh` 编译和执行。默认测试方阵 `N=64,128,…,1280`，也可以指定单个 N（64～1280，且为 64 的倍数）。

### CPU 计时

- 先预热执行 1 次；
- 再执行 5 次；
- 使用 5 次平均耗时计算 GFLOPS；
- 只统计 CPU GEMM，不包含数据生成、结果校验等时间。

### GPU 计时

每个 GPU 版本先预热 1 次，再进行 5 次完整端到端测试并取平均值。每次测试包含：

1. `cudaMalloc` 分配 A、B、C；
2. A、B 从主机复制到 GPU（H2D）；
3. 执行 GPU kernel 或 cuBLAS GEMM；
4. C 从 GPU 复制回主机（D2H）；
5. `cudaFree` 释放显存。

计算公式为：

```text
GFLOPS = 2 × N³ / 平均耗时（秒）/ 10⁹
```

输出格式为：

```text
N GFLOPS 最大绝对误差
```

因此，CPU 与 GPU 的结果可以按端到端口径直接比较；GPU 各版本之间也必须使用相同 GPU、相同 N 和相同计时方式比较。

### 运行命令

在可访问 GPU 的普通 Terminal 中执行：

```bash
cd /home/hill/gemm
LD_LIBRARY_PATH=/usr/lib64:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-} \
  ./cuda/run_benchmark.sh 1024
```

不指定 N 时，测试全部规模：

```bash
./cuda/run_benchmark.sh
```

脚本会自动识别 GPU 架构，编译 `cuda/gpu_runner.cu`（其中包含各版本 `.cu` 实现）生成基准程序；新增版本时，将其实现纳入 `gpu_runner.cu` 即可，无需修改脚本。在项目虚拟环境安装 matplotlib 时还会生成性能图。结果保存在：

- RTX 4090 D：`fig-gpu-4090/gpu_results.m`、`gpu_comparison.png`；
- A100：`fig-gpu-a100/gpu_results.m`、`gpu_comparison.png`。

仅根据已有结果重绘图表：

```bash
.venv/bin/python cuda/plot_results.py \
  fig-gpu-4090/gpu_results.m fig-gpu-4090/gpu_comparison.png
```

`cuda/plot_results.py` 只解析结果文件并绘图，不编译、不运行 GPU，也不会重新测量。

## 5. N=1024 实测结果

以下结果来自 RTX 4090 D 的 [gpu_results.m](../fig-gpu-4090/gpu_results.m)，采用上述端到端计时口径。GFLOPS 越高越好；误差为相对 CPU 参考结果的最大绝对误差。

| 顺序 | 版本 | GFLOPS | 相对 GPU naive | 最大绝对误差 |
| ---: | --- | ---: | ---: | ---: |
| 1 | `cpu-reference` | 0.279678 | — | 0 |
| 2 | `cuda-naive` | 461.917 | 1.000× | 3.55e-14 |
| 3 | `cuda-naive-ilp4` | 459.575 | 0.995× | 9.95e-14 |
| 4 | `cuda-naive-1x2` | 463.036 | 1.002× | 3.55e-14 |
| 5 | `cuda-tiled-32` | 455.476 | 0.986× | 3.55e-14 |
| 6 | `cuda-reg-2x2` | 464.389 | 1.005× | 3.55e-14 |
| 7 | `cuda-reg-4x4` | 465.018 | **1.007×** | 3.55e-14 |
| 8 | `cuda-reg-4x4-64` | 407.438 | 0.882× | 3.55e-14 |
| 9 | `cublas-dgemm` | 412.719 | 0.894× | 3.55e-14 |

本次端到端测试中，`cuda-reg-4x4` 最快，比 GPU naive 快约 0.7%。CPU reference 的 GFLOPS 明显低于 GPU，但它是普通三重循环版本，不代表 CPU AVX2 优化版本的性能。


