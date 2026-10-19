# Nsight 分析本项目 CUDA GEMM（RTX 4090 D）

## 1. 分工与环境

- **Nsight Systems (`nsys`)**：先看全程序时间线；区分 CPU 参考计算、CUDA API、内存拷贝、同步、手写 kernel 与 cuBLAS kernel 的 GPU 执行时间。
- **Nsight Compute (`ncu`)**：再选一个具体 GPU kernel，看计算管线、访存、占用率、warp 等硬件指标。不要一开始对全程序的所有 kernel 运行 `--set full`。
- 本环境：RTX 4090 D，计算能力 8.9；`ncu` 2022.4.1，`nsys` 2022.4.2，`/usr/bin/nvcc` 为 CUDA 12.0。驱动显示的 CUDA 13.2 是**驱动支持的最高 CUDA 版本**，不等于本机编译器版本。
- 下面的命令请在**能正常运行 CUDA 的普通 Terminal** 中执行；若 Trae 代理沙箱禁止访问 `/dev/nvidia*` 或 `/proc`，换成普通 Terminal。分析报告先写到 `/tmp`，不覆盖 `fig-gpu-4090/` 中的基准数据。当前环境有旧版 535 的 `libcuda.so.1`，已验证必须在运行命令前优先指定 `/usr/lib64` 中与 595 驱动匹配的库。Nsight 是否有 GPU profiling 权限，仍须现场检查。

## 2. 编译并确认基准

在项目目录编译独立的 profiling 二进制（脚本中的可执行文件是临时文件，运行后会被删除，不要假定 `/tmp/gpu_gemm_4090_profile.x` 已存在）。先检查 Nsight 是否支持当前设备、报告集合和 CUDA 基准能否运行：

```bash
cd /home/hill/gemm
/usr/bin/nvcc -O3 -arch=sm_89 -lineinfo -I/usr/include gpu_gemm.cu \
  -L/usr/lib/x86_64-linux-gnu -lcublas -lcudart -o /tmp/gpu_gemm_4090_profile.x
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
ncu --version
nsys --version
ncu --list-sets
nsys status -e
export LD_LIBRARY_PATH=/usr/lib64:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
/tmp/gpu_gemm_4090_profile.x > /tmp/gemm_4090_profile_results.m
```

`-lineinfo` 方便将指标关联回 CUDA 源码，通常不改变优化级别；普通性能测量仍以**不挂 profiler** 的 `gpu_results.m` 为准。这里的 `export` 只影响当前 shell，后续 `nsys` / `ncu` 启动的进程也会继承；请在**同一个 Terminal 会话**执行后续命令。只要仍报 `unsupported display driver / cuda driver combination`，先检查库路径和实际加载的 `libcuda.so.1`，不要直接升级 Toolkit。上述验证运行会执行全部 120 组实验及 CPU 参考计算，输出仅写入 `/tmp`。`ncu --list-sets` 本机列有 `default` 和 `detailed`；这不能代替在 GPU 上采集成功的验证。

基准源文件 `gpu_gemm.cu` 对 64～1280（步长 64）的每个规模、每个版本做 CPU 参考计算；每次 GPU 计时为一次预热，接着用 CUDA event 计 5 次调用的平均值。矩阵分配、CPU 参考计算、H2D/D2H 拷贝和首次预热均不计入报告的 GFLOPS。**Nsight 跟踪整个程序可能明显慢于原始测试**，尤其是 `ncu` 会多次重放选中的 kernel；不要把 profiler 下的端到端时间当作原始性能。

## 3. 先用 Nsight Systems 看时间线

### 命令行基本用法

`nsys` 是 Nsight Systems 的命令行工具。常用命令包括：

- `nsys profile`：运行程序并采集性能数据，生成 `.nsys-rep` 报告。
- `nsys launch`：启动一个准备好接受分析的程序。
- `nsys start` / `nsys stop`：开始或停止一次性能分析会话，并保存采集结果。
- `nsys cancel`：取消当前会话并丢弃已采集的数据。
- `nsys service`：启动 Nsight Systems 数据服务。
- `nsys stats`：从 `.nsys-rep` 或 SQLite 文件生成统计信息。
- `nsys status`：检查命令行工具和采集环境状态。
- `nsys shutdown`：断开已启动进程与分析器的连接并关闭分析会话。
- `nsys sessions list`：列出当前活动的分析会话。
- `nsys export`：将 `.nsys-rep` 报告导出为其他格式。
- `nsys analyze`：分析报告或 SQLite 文件，查找潜在优化机会。
- `nsys recipe`：运行面向多节点分析的 recipe。
- `nsys nvprof`：将 `nvprof` 参数转换为 `nsys` 参数并执行采集。

查看总帮助或指定子命令的帮助：

```bash
nsys --help
nsys --help profile
```

最简单的采集方式是 `nsys profile ./my-application`。本项目使用下面针对 CUDA、cuBLAS 和 NVTX 的采集命令：

```bash
cd /home/hill/gemm
nsys profile --trace=cuda,cublas,nvtx --sample=none --cpuctxsw=none \
  --output=/tmp/gemm_4090_nsys /tmp/gpu_gemm_4090_profile.x
nsys stats --report cudaapisum --report gpukernsum --report gpumemtimesum \
  /tmp/gemm_4090_nsys.nsys-rep
nsys stats --report gputrace /tmp/gemm_4090_nsys.nsys-rep
```

生成 `/tmp/gemm_4090_nsys.nsys-rep`；再次运行同名输出前先换一个 `--output` 路径，或确认可以覆盖该报告。可用图形界面 `nsys-ui /tmp/gemm_4090_nsys.nsys-rep` 打开（远程终端需图形桌面/转发）。`gpukernsum` 汇总所有规模；`gputrace` 按启动列出 GPU 活动，结合时间线定位同一规模的实际 kernel；若本机未提供该 report，请用 `nsys stats --help-reports` 核对。在时间线中对照 CUDA API 发起、GPU kernel 执行与 memcpy，重点检查：

1. cuBLAS 调用是否启动了与手写 kernel **不同的 GPU kernel**、各次调用实际耗时是多少；不要把 CPU 侧 `cublasDgemm` API 调用耗时直接当作 GPU 计算时间。
2. GPU 调用之间是否有大的空白；本程序逐个尺寸计算 CPU 参考值并同步，整段时间线会有空白，这**不能直接推出** GEMM kernel 自身性能差。
3. 同一尺寸的 5 次计时是否稳定，是否存在首次运行/选算法开销；要隔离尺寸，最好让程序只运行一个规模和一个版本，并保留预热及多次测量（当前程序没有这种筛选参数）。
4. `gpu_results.m` 中 cuBLAS 和手写 kernel 的 GFLOPS 基于相同的 `2n³ / 时间`，但不同版本分别计时；`nsys stats` 对**所有尺寸**的汇总值不能直接与某个 `n=1024` 的 GFLOPS 对比。

`nsys` 2022.4 的 `--gpu-metrics-device` 可选，但不是初次分析的必要条件；先确保 CUDA 时间线正常，避免采集权限/设备支持问题干扰。

<br />

## 4. 再用 Nsight Compute 分析单个 kernel

### 命令行基本用法

`ncu` 是 Nsight Compute 的命令行工具，用于分析单个 CUDA kernel 的硬件性能指标。常用命令和参数包括：

- `ncu --version`：查看 Nsight Compute 的版本。
- `ncu --help`：查看命令行帮助；需要查看具体参数时，可使用 `ncu --help <选项>` 或直接搜索帮助输出。
- `ncu --list-sets`：列出当前安装版本支持的指标集合，例如 `default`、`detailed` 等。
- `ncu --set <集合名>`：选择一次采集使用的指标集合。通常先使用 `default`，需要更多指标时再使用 `detailed`。
- `--kernel-name <名称或正则表达式>`：按 kernel 名称过滤采集目标，`regex:` 前缀表示使用正则表达式。
- `--launch-skip <次数>`：跳过目标 kernel 前面的若干次启动。
- `--launch-count <次数>`：限制实际采集的 kernel 启动次数。
- `--export <文件名>`：将采集结果保存为 `.ncu-rep` 报告。
- `ncu --import <报告文件>`：打开已经保存的 Nsight Compute 报告。

查看帮助和可用指标集合：

```bash
ncu --help
ncu --version
ncu --list-sets
```

`ncu` 应优先针对一个明确的 kernel 和一次启动进行采集，不建议一开始对整个程序的所有 kernel 使用完整指标集合。采集过程可能需要多次重放 kernel，并且会改变缓存、时钟和执行时间；因此，Nsight Compute 报告用于定位计算、访存和占用率瓶颈，正式性能结果仍应以未挂 profiler 的重复基准测试为准。

先只分析一个手写 kernel。当前程序每个版本从 n=64 起，步长 64；每个尺寸启动同名 kernel **一次预热 + 五次计时**。因此对 `gemm_reg_2x2`，n=1024 是第 16 个尺寸：跳过前 15×6=90 次**匹配该名称**的启动，再采集该尺寸的第一次启动（预热）。若修改了程序的测试规模、循环或 kernel 数，请重新计算，不要沿用 90。

```bash
cd /home/hill/gemm
ncu --set default --kernel-name 'regex:gemm_reg_2x2' \
  --launch-skip 90 --launch-count 1 \
  --export /tmp/gemm_4090_reg2x2_n1024 /tmp/gpu_gemm_4090_profile.x
ncu --import /tmp/gemm_4090_reg2x2_n1024.ncu-rep
```

如果想采集 n=1024 的**第一次计时调用**而非预热，将 `--launch-skip 90` 改为 `--launch-skip 91`。核对报告中的 kernel 名称、grid/block 和输入规模；该 kernel 本身无 n=1024 标签，不能仅凭名称确认匹配。`--launch-count 1` 只限制采集次数，不会让程序跳过其他尺寸的 CPU 参考计算。`ncu` 的 kernel replay 可能改变缓存和时钟，因此报告用来解释瓶颈，不作为未挂 profiler 的 GFLOPS。

需要深入分析同一尺寸时再采集 `detailed`（可多次 replay）：

```bash
ncu --set detailed --kernel-name 'regex:gemm_reg_2x2' \
  --launch-skip 91 --launch-count 1 \
  --export /tmp/gemm_4090_reg2x2_n1024_detail /tmp/gpu_gemm_4090_profile.x
ncu --import /tmp/gemm_4090_reg2x2_n1024_detail.ncu-rep
```

对 tiled / reg 4x4，可将过滤名称替换为实际 kernel 名，并同样计算跳过次数。对 cuBLAS，请先从 `nsys` 时间线定位 n=1024 的实际 GPU kernel 名、启动次数及是否有辅助 kernel；cuBLAS 名称和每次调用的 kernel 数不保证固定，**不能直接套用** **`--launch-skip 91`**。建议为同一尺寸/版本添加基准筛选入口后再精确采集，或者在 `nsys` 时间线逐次核对对应启动；当前文档不把尚未实现的筛选参数写成可运行命令。

详细报告里先看 `SpeedOfLight`、`ComputeWorkloadAnalysis`、`MemoryWorkloadAnalysis`、`Occupancy`、`WarpStateStats`。重点验证：

- GPU 运行时长与时钟是否稳定；FP64 计算管线是否接近瓶颈（查双精度指令及相关吞吐指标）；是否出现 tensor 指令，以及 cuBLAS 实际采用的算法。不要单凭 GFLOPS 推定它走了 FP64 Tensor Core。
- DRAM/L2 带宽是否接近上限；低 occupancy、shared-memory bank conflict、warp 等待原因是否比 FP64 吞吐更值得优化。
- 以相同矩阵尺寸比较 naive、tiled、reg 与 cuBLAS 的 GPU 时长和工作量；重点区分 FP64 算力上限、naive 的 L2 缓存复用、tiled 的共享内存同步成本和 reg 的寄存器/occupancy 影响。报告中的瓶颈提示是线索，不是自动证明；Nsight Compute 重放与采集会扰动时序，绝对性能应回到**未挂 profiler** 的多次重复测试。

如果出现 `ERR_NVGPUCTRPERM`，说明硬件计数器权限不足，需要管理员允许 profiling（例如配置 NVIDIA 驱动的 profiling 权限），不要误判为 CUDA kernel 错误；若出现设备或指标不支持，先确认这套 2022.4 工具能否支持 Ada（sm\_89），再考虑更新 Nsight 版本。`nsys` 时间线和 `ncu` 硬件计数器对权限的要求可能不同。

A100 与 RTX 4090 D 的规格、实测数据及性能差异分析见 [GPU 显卡性能规格](GPU显卡性能规格.md)；本文只保留 Nsight 的操作与验证流程。
