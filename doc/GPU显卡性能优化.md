# GPU 显卡性能优化：以 naive GEMM 为基线

## 1. 目标与测量口径

本项目的 `gpu_gemm.cu` 实现双精度矩阵乘法 `C = A × B`。`gemm_naive`（第 24～32 行）每个线程负责一个输出元素：沿 K 维读取 A 的一行和 B 的一列，累加到 `sum` 后写回 C。相邻线程访问 B 的相邻列，有利于合并访存；但跨线程的 A/B 数据复用主要依赖缓存，尚未通过共享内存显式复用。是否真的受访存限制，必须用 profiler 测量，不能仅凭代码推断。

先明确两个不同目标：

- **Kernel-only 性能**：`time_kernel`（第 220～236 行）预热一次，CUDA Event 计时后续 5 次调用，取平均值；不包含分配、主机与设备拷贝及 CPU 参考计算。`run_mode`（第 259～284 行）以 `2N³ / 时间` 换算 GFLOP/s，并输出与 CPU 参考结果的最大绝对误差。
- **端到端延迟**：还须计算分配、数据传输、同步及 CPU 侧开销。不能把全程序时间线时长直接与 kernel-only GFLOP/s 比较。

固定同一 GPU、`double` 精度、编译选项、矩阵尺寸、计时方式与误差标准，比较同一 N 的耗时；多次重复并关注波动。未挂 profiler 的 CUDA Event 结果作为最终性能比较依据，Nsight 用来定位瓶颈。

## 2. 先建立 naive 基线

1. 编译带 `-lineinfo` 的 profiling 可执行文件；确认运行环境可访问 GPU、动态库与驱动匹配。具体编译命令、`LD_LIBRARY_PATH` 及环境检查参见 [Nsight 使用说明](Nsight使用说明.md)。
2. 保存 naive 在各 N 的 GPU 时间、GFLOP/s 和最大误差；选择一个具有代表性的固定 N（例如 1024）深入分析。其他实现需使用相同输入、精度和计时口径。
3. 当前程序按 N=64～1280、步长 64 运行六个版本，包含 CPU 参考计算；尚无单独筛选 N/版本的命令行参数。全程序报告的同名 kernel 汇总混合了不同尺寸，**不能**直接当成 N=1024 的耗时。

## 3. Nsight Systems：判断时间花在哪里

RTX 4090 D 上已成功生成有效的 `/tmp/gemm_4090_nsys_fixed.nsys-rep`，且已确认报告包含 CUDA API 与 GPU kernel 数据。它位于临时目录，长期留存前应另行保存。使用生成报告时对应的 Nsight Systems 版本读取：

```bash
/tmp/nsys-2024-local/opt/nvidia/nsight-systems/2024.2.3/bin/nsys stats \
  --report cuda_api_sum --report cuda_gpu_kern_sum \
  /tmp/gemm_4090_nsys_fixed.nsys-rep
```

如需重新采集，在**可访问 GPU 的普通 Terminal** 中运行（不要把受限沙箱里生成、没有 CUDA 记录的报告当作分析依据）：

```bash
/tmp/nsys-2024-local/opt/nvidia/nsight-systems/2024.2.3/bin/nsys profile \
  --trace=cuda,cublas,nvtx --sample=none --cpuctxsw=none \
  --output=/tmp/gemm_4090_nsys_next \
  /tmp/gpu_gemm_4090_profile.x
```

在时间线中分别看 CPU 参考计算、H2D/D2H 拷贝、CUDA API、同步、GPU kernel 执行与 GPU 空闲间隔。特别区分 CPU 侧 API 调用时间和 GPU 实际执行时间。若 API/传输占主要成本，优先考虑端到端数据管理与减少启动开销；若 naive 的 GPU 执行占主要成本，再进入内核分析。报告汇总适合确定总体热点，但比较单个尺寸必须在时间线上定位对应调用，或为基准增加单尺寸筛选后重新采集。

> 本机系统自带的旧版 `nsys` 缺少 QDSTRM 导入器；上述 2024.2.3 版本已成功生成报告。不要将只得到 `.qdstrm` 或不含 CUDA 记录的 `.nsys-rep` 误认为有效分析结果。临时安装目录和 `/tmp` 文件重启或清理后可能消失。

## 4. Nsight Compute：判断 naive kernel 为什么慢

对固定 N 的 **单次** naive kernel 使用 `ncu`，先采集默认指标，再按需查看 `SpeedOfLight`、`ComputeWorkloadAnalysis`、`MemoryWorkloadAnalysis`、`Occupancy` 和 `WarpStateStats`。当前程序对每个尺寸发起一次预热加五次计时；N=1024 是第 16 个尺寸，若仅匹配 `gemm_naive`，跳过前 `15 × 6 = 90` 次启动，跳过 91 次则采集该尺寸首次计时调用：

```bash
ncu --set default --kernel-name 'regex:gemm_naive' \
  --launch-skip 91 --launch-count 1 \
  --export /tmp/gemm_4090_naive_n1024 \
  /tmp/gpu_gemm_4090_profile.x
ncu --import /tmp/gemm_4090_naive_n1024.ncu-rep
```

核对报告中的 kernel 名称、grid/block 与预期尺寸；如修改循环、尺寸或启动次数，必须重新计算 `--launch-skip`。`ncu` 可能重放 kernel，采样时长不宜当作最终 GFLOP/s。若提示 `ERR_NVGPUCTRPERM`，应检查硬件计数器权限；若设备/指标不支持，检查工具版本与计算能力兼容性。详见 [Nsight 使用说明](Nsight使用说明.md)。

优先回答以下问题，然后再选优化方向：

| 证据 | 可能方向 | 避免误判 |
|---|---|---|
| DRAM/L2 流量及吞吐高、重复读取明显 | 分块并显式复用 A/B | 缓存也会提供复用，不能仅据源码认定 DRAM 瓶颈 |
| FP64 计算管线接近饱和 | 评估硬件 FP64 上限及指令效率 | 4090 D 的 FP32 峰值不是 DGEMM 峰值 |
| 寄存器或共享内存限制 occupancy | 减小 tile、调整线程/每线程输出数 | occupancy 高不等于一定更快 |
| 同步、依赖等待或启动开销明显 | 调整 tile/K 分块或减少小任务固定开销 | 不同 N 的主要瓶颈可能不同 |

## 5. 按证据逐项优化

| 版本/方法 | 核心做法 | 重点验证 |
|---|---|---|
| `cuda-naive` | 每线程计算 1 个 C 元素 | 作为正确性和耗时基线 |
| `cuda-tiled-32`（第 34～51 行） | 32×32 分块，A/B 搬入共享内存，块内复用 | 全局读取是否下降；同步及共享内存成本 |
| `cuda-reg-2x2`（第 53～84 行） | 每线程计算 2×2 个输出，寄存器累加 | 复用收益与寄存器占用 |
| `cuda-reg-4x4`（第 86～121 行） | 每线程计算 4×4 个输出 | 指令/寄存器压力与 occupancy |
| `cuda-reg-4x4-64`（第 123～169 行） | 更大输出 tile，K 维按 8 分块 | 对不同 N 的可用并发、边界与吞吐影响 |
| `cublas-dgemm`（第 238～257 行） | 作为成熟库实现的参照 | 观察实际 GPU kernel，不预设其在每个规模都最快 |

还可在确认瓶颈后分别试验线程块布局、K 分块长度、合并访存与共享内存 bank conflict、双缓冲/预取等；每次只改变一个因素，并复测正确性、同尺寸时长、吞吐及 profiler 指标。不要一次叠加多项技术而无法归因。不同 GPU 架构、FP64 性能与矩阵规模下，最佳配置可能不同；也不要未经精度许可就把 FP64 改成 TF32/FP16。

## 6. 复测与结论记录

推荐以 `GPU / 编译选项 / N / 实现 / 最大误差 / 未挂 profiler 的平均 GPU 时间 / GFLOP/s / Nsight 证据` 逐行记录，计算同尺寸加速比 `T_naive / T_optimized`。先检查结果正确，再判断优化是否有效；必要时同时记录端到端延迟，避免仅提高 kernel 吞吐却没有改善实际任务。

已有 RTX 4090 D 数据显示部分规模的 naive 快于当前 tiled/寄存器实现，说明共享内存或寄存器分块**并非必然更快**。先用 Nsight 确认瓶颈，再用未挂 profiler 的同条件实验验证改善；A100 与 4090 D 的 FP64 规格和实测差异见 [GPU 显卡性能规格](GPU显卡性能规格.md)。
