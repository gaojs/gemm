# Nsight 分析本项目 CUDA GEMM（RTX 4090 D）

## 1. 分工与环境

- **Nsight Systems (`nsys`)**：先看全程序时间线；区分 CPU 参考计算、CUDA API、内存拷贝、同步、手写 kernel 与 cuBLAS kernel 的 GPU 执行时间。
- **Nsight Compute (`ncu`)**：再选一个具体 GPU kernel，看计算管线、访存、占用率、warp 等硬件指标。不要一开始对全程序的所有 kernel 运行 `--set full`。
- 本环境：RTX 4090 D，计算能力 8.9；`ncu` 2022.4.1，`nsys` 2022.4.2，`/usr/bin/nvcc` 为 CUDA 12.0。驱动显示的 CUDA 13.2 是**驱动支持的最高 CUDA 版本**，不等于本机编译器版本。
- 下面的命令请在**能正常运行 CUDA 的普通 Terminal** 中执行；若 Trae 代理沙箱禁止访问 `/dev/nvidia*` 或 `/proc`，换成普通 Terminal。分析报告建议先写到 `/tmp`，不要误覆盖已有基准数据。

## 2. 编译并确认基准

在项目目录执行；若已经有通过 `-arch=sm_89` 编译的 `/tmp/gpu_gemm_4090.x`，可跳过编译。

```bash
cd /home/hill/gemm
/usr/bin/nvcc -O3 -arch=sm_89 -lineinfo -I/usr/include gpu_gemm.cu \
  -L/usr/lib/x86_64-linux-gnu -lcublas -lcudart -o /tmp/gpu_gemm_4090_profile.x
nvidia-smi
ncu --version
nsys --version
```

`-lineinfo` 方便将指标关联回 CUDA 源码，通常不改变优化级别；普通性能测量仍以**不挂 profiler** 的 `gpu_results.m` 为准。若运行时找不到 CUDA 动态库，先确认 `ldd /tmp/gpu_gemm_4090_profile.x` 的输出，再按实际库路径设置 `LD_LIBRARY_PATH`（本环境可尝试 `/usr/lib64:/usr/lib/x86_64-linux-gnu`）。

基准源文件 `gpu_gemm.cu` 对 64～1280（步长 64）的每个规模、每个版本做 CPU 参考计算；每次 GPU 计时为一次预热，接着用 CUDA event 计 5 次调用的平均值。矩阵分配、CPU 参考计算、H2D/D2H 拷贝和首次预热均不计入报告的 GFLOPS。**Nsight 跟踪整个程序可能明显慢于原始测试**，尤其是 `ncu` 会多次重放选中的 kernel；不要把 profiler 下的端到端时间当作原始性能。

## 3. 先用 Nsight Systems 看时间线

```bash
cd /home/hill/gemm
nsys profile --trace=cuda,cublas,nvtx --sample=none --cpuctxsw=none \
  --output=/tmp/gemm_4090_nsys /tmp/gpu_gemm_4090_profile.x
nsys stats --report cudaapisum --report gpukernsum --report gpumemtimesum \
  /tmp/gemm_4090_nsys.nsys-rep
```

生成 `/tmp/gemm_4090_nsys.nsys-rep`；可用图形界面 `nsys-ui /tmp/gemm_4090_nsys.nsys-rep` 打开（远程终端需图形桌面/转发）。在时间线中对照 CUDA API 发起、GPU kernel 执行与 memcpy，重点检查：

1. cuBLAS 调用是否启动了与手写 kernel **不同的 GPU kernel**、各次调用实际耗时是多少；不要把 CPU 侧 `cublasDgemm` API 调用耗时直接当作 GPU 计算时间。
2. GPU 调用之间是否有大的空白；本程序逐个尺寸计算 CPU 参考值并同步，整段时间线会有空白，这**不能直接推出** GEMM kernel 自身性能差。
3. 同一尺寸的 5 次计时是否稳定，是否存在首次运行/选算法开销；要隔离尺寸，最好让程序只运行一个规模和一个版本，并保留预热及多次测量（当前程序没有这种筛选参数）。
4. `gpu_results.m` 中 cuBLAS 和手写 kernel 的 GFLOPS 基于相同的 `2n³ / 时间`，但不同版本分别计时；`nsys stats` 对**所有尺寸**的汇总值不能直接与某个 `n=1024` 的 GFLOPS 对比。

`nsys` 2022.4 的 `--gpu-metrics-device` 可选，但不是初次分析的必要条件；先确保 CUDA 时间线正常，避免采集权限/设备支持问题干扰。

## 4. 再用 Nsight Compute 分析单个 kernel

先对手写的 `gemm_reg_2x2` 采集一个匹配的 launch；**注意这取的是首次匹配（通常 n=64）**，不能用来解释 n=1024 的数值：

```bash
cd /home/hill/gemm
ncu --set default --kernel-name 'regex:gemm_reg_2x2' --launch-count 1 \
  --export /tmp/gemm_4090_reg2x2 /tmp/gpu_gemm_4090_profile.x
ncu --import /tmp/gemm_4090_reg2x2.ncu-rep
```

实际排查 `n=1024` 时，建议给基准程序增加“只运行指定版本和 n=1024”的命令行筛选，然后再用同样的 `ncu` 命令；否则 `ncu` 的 kernel 过滤只改变**采集范围**，不会跳过程序中昂贵的 CPU 参考计算和其余测试。对 cuBLAS，应先从 `nsys` 的 GPU timeline 找到真实 kernel 名称，再以相应的 `--kernel-name 'regex:...'` 过滤；cuBLAS 内部名称随版本和形状变化，不要猜测它一定叫 `cublasDgemm`。可用 `ncu --kernel-name ... --launch-skip N --launch-count 1` 跳过前 N 次**匹配**的启动；使用前先核对实际次数与尺寸。

需要更深入时，选定同一尺寸的手写 kernel 和 cuBLAS kernel 分别采集；`ncu --list-sections` 可列出本机支持的 section：

```bash
ncu --set detailed --kernel-name 'regex:gemm_reg_2x2' --launch-count 1 \
  --export /tmp/gemm_4090_reg2x2_detail /tmp/gpu_gemm_4090_profile.x
ncu --import /tmp/gemm_4090_reg2x2_detail.ncu-rep
```

详细报告里先看 `SpeedOfLight`、`ComputeWorkloadAnalysis`、`MemoryWorkloadAnalysis`、`Occupancy`、`WarpStateStats`。重点验证：

- GPU 运行时长与时钟是否稳定；FP64 计算管线是否接近瓶颈（查双精度指令及相关吞吐指标）；是否出现 tensor 指令，以及 cuBLAS 实际采用的算法。不要单凭 GFLOPS 推定它走了 FP64 Tensor Core。
- DRAM/L2 带宽是否接近上限；低 occupancy、shared-memory bank conflict、warp 等待原因是否比 FP64 吞吐更值得优化。
- 以相同矩阵尺寸比较两种 kernel 的 GPU 时长和工作量；Nsight Compute 重放与采集会扰动时序，绝对性能应回到**未挂 profiler** 的多次重复测试。

如果出现 `ERR_NVGPUCTRPERM`，说明硬件计数器权限不足，需要管理员允许 profiling（例如配置 NVIDIA 驱动的 profiling 权限），不要误判为 CUDA kernel 错误；若出现设备或指标不支持，先确认这套 2022.4 工具能否支持 Ada（sm_89），再考虑更新 Nsight 版本。`nsys` 时间线和 `ncu` 硬件计数器对权限的要求可能不同。

## 5. 为什么 A100 上 cuBLAS 一骑绝尘，而 4090 D 上不是？

这组测试是 **FP64（`double`）DGEMM**，不是 FP32/TF32。A100 面向 HPC，具有强大的 FP64 计算能力与 FP64 Tensor Core；GeForce RTX 4090 D 面向消费级图形和低精度 AI，其原生 FP64 吞吐相对于 FP32 被大幅限制。4090 D 的 FP32/Tensor Core 宣传峰值**不能**当作 DGEMM 峰值。因此 A100 的 cuBLAS 可以利用硬件/优化路径显著拉开差距；4090 D 上多个实现接近其 FP64 吞吐约束，cuBLAS 的算法选择、访存与 kernel 固定开销可能使其在本项目的小到中等矩阵规模下不占优。A100 的 cuBLAS 是否实际用了 FP64 Tensor Core、4090 D 的具体瓶颈是什么，还需以上述 profiler 实测证明，不能只根据结果文件断定。

| 环境 / n=1024 | cuBLAS DGEMM | 最快的手写版本 | 关系 |
|---|---:|---:|---:|
| A100 SXM4 | 13498 GFLOPS | reg-2x2：3918 GFLOPS | cuBLAS 约 3.45 倍 |
| RTX 4090 D | 902 GFLOPS | reg-2x2：1130 GFLOPS | cuBLAS 约为 0.80 倍 |

A100 原始数据见 `fig-gpu-a100/gpu_results.m`，4090 D 数据见 `fig-gpu-4090/gpu_results.m`。4090 D 下 reg-2x2 的最好**单个规模**为 n=1024 的约 1130 GFLOPS，cuBLAS 最好**单个规模**为 n=960 的约 989 GFLOPS；上表固定 n=1024，避免用不同尺寸的峰值相除。

不能把 4090 D 的差异仅归咎于 CUDA 12.0 或驱动：当前数据只能说明**该硬件、软件栈、矩阵规模与算法选择**下的表现；CUDA 版本可能影响算法选择，但并无直接证据说明它是主要原因。基准固定调用 `cublasDgemm`，没有指定特殊的混合精度数学模式；单次测试、GPU 动态频率和数据复用也会影响结果。要确定贡献，按相同编译参数、相同 n、相同计时口径复测，同时记录 `nvidia-smi` 时钟和功耗，并在 Nsight 中检查选中的 kernel、FP64/Tensor 指令与实际执行时间。
