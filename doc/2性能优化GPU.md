# GPU GEMM 优化复现记录

## 1. 目标

在 NVIDIA A100-SXM4-80GB GPU 上复现并优化本项目的 `C = A × B` 双精度 GEMM。CPU 版本通过 `MY_MMult` 接口测试；GPU 版本单独建立 CUDA 基准，避免改变原有 CPU 构建流程。

## 2. 已实现的 GPU 方法

### 2.1 CUDA 朴素 kernel：`cuda-naive`

每个 CUDA thread 负责计算 C 的一个元素，沿 K 维循环累加。该版本结构简单，适合作为 GPU 优化起点，但每个线程会重复从全局内存读取 A、B 数据。

### 2.2 共享内存分块：`cuda-tiled-32`

使用 32×32 tile。线程块协同将 A、B 的 tile 加载到 shared memory，再进行内积计算，从而提高数据复用并减少全局内存访问。边界矩阵使用零填充，支持 64 到 1280 的非 tile 整数倍尺寸。

### 2.3 cuBLAS：`cublas-dgemm`

调用 NVIDIA 官方 cuBLAS `cublasDgemm`，作为高性能库参考。cuBLAS 通常还会使用针对 A100 优化的 kernel、调度和内存访问策略。

## 3. 文件和运行方式

- `gpu_gemm.cu`：CUDA 基准程序
- `gpu_run_benchmark.sh`：编译、运行和绘图脚本
- `gpu_plot_results.py`：GPU 结果绘图脚本
- `fig-gpu/`：GPU 结果目录

`cicc` 是 `nvcc` 使用的 CUDA NVVM 内部设备编译器，不需要单独加入 `PATH`。本项目完整 CUDA 13.2 工具链中的实际路径是 `/usr/local/cuda/nvvm/bin/cicc`；`whereis cicc` 查不到并不代表文件不存在，可直接检查：

```bash
test -x /usr/local/cuda/nvvm/bin/cicc && echo "cicc OK"
```

在 CUDA 运行时正常的环境中执行：

```bash
cd /home/hill/gemm
./gpu_run_benchmark.sh
```

程序会测试矩阵规模 64、128、……、1280，输出每个版本的 GFLOPS 和最大误差，并生成：

```text
fig-gpu/gpu_results.m
fig-gpu/gpu_comparison.png
```

## 4. 当前环境与运行说明

当前测试环境为 NVIDIA A100-SXM4-80GB，计算能力 8.0，使用 CUDA Toolkit 13.2。项目脚本固定使用完整工具链 `/usr/local/cuda/bin/nvcc`、对应的 CUDA 头文件和库，并使用项目 `.venv` 绘图。

`cicc` 不需要加入 `PATH`。它位于 `/usr/local/cuda/nvvm/bin/cicc`，由同一套 CUDA 13.2 的 `nvcc` 自动调用；`whereis cicc` 查不到不影响编译。只有当该文件实际不存在，或 `nvcc` 报 `cicc: No such file or directory` 时，才说明 Toolkit 安装不完整。

Trae 执行沙箱不能访问 `/dev/nvidia*` 时会在 `cudaGetDeviceProperties()` 处失败；GPU benchmark 应在普通 Terminal 中运行。

## 5. 已修复的问题

- `cuda-tiled-32` 已使用 32×32 线程块，与 shared-memory tile 匹配。
- 结果文件使用真正的换行符，绘图脚本可以解析 `gpu_results.m`。
- benchmark 失败时使用临时文件，不会覆盖已有有效结果。
- A100 实测三个版本均通过正确性校验，最大误差约为 `1e-14`。

## 6. 正确性与性能判定

GPU 结果必须同时满足：

- 最大误差处于双精度浮点计算的正常范围；
- tiled kernel 性能通常高于 naive kernel；
- cuBLAS DGEMM 作为 A100 高性能参考；
- 只有运行完成并生成 `fig-gpu/gpu_results.m` 后，才记录 GFLOPS 和优化结论。

第 7 节记录了 2026-09-07 在 A100 上的真实运行结果（数据保存在 `fig-gpu/`，见下）。

## 7. A100 运行结果（2026-09-07）

在无沙箱限制的 Terminal 中执行 `./gpu_run_benchmark.sh` 后，GPU GEMM 基准成功完成，结果已写入：

- `fig-gpu/gpu_results.m`
- `fig-gpu/gpu_comparison.png`

三个版本的最大误差均为 `1e-14` 量级，正确性校验通过。

| 版本 | 峰值 GFLOPS（n=1280 附近） | 说明 |
|------|------------------|------|
| `cuda-naive` | ≈ 1.86e3 | 每线程算一个 C 元素，全局内存反复读取 |
| `cuda-tiled-32` | ≈ 2.30e3 | 共享内存 32×32 分块，数据复用提升 ~24% |
| `cublas-dgemm` | ≈ 1.57e4 | NVIDIA 官方库，远高于手写 kernel |

## 8. 进一步优化方向

`cublas-dgemm` 是本项目当前最快、也是最可靠的实现，但不是 GPU 理论峰值。它已经由 NVIDIA 针对 A100 做了高度优化；手写 kernel 仍可作为学习和对比对象继续优化：

- 使用二维线程块、寄存器分块和向量化加载，减少 shared memory 和全局内存访问。
- 对 A、B 做 shared-memory staging 或分块预取，尝试双缓冲隐藏访存延迟。
- 调整 tile 尺寸、线程块形状和循环展开参数，并用 Nsight Compute 分析占用率、访存和计算吞吐。
- 针对固定尺寸矩阵使用专用 kernel；若允许改变数据布局，可使用矩阵转置或 packing 改善访存合并。
- 对半精度、TF32 或混合精度问题，可进一步使用 Tensor Core；当前双精度 DGEMM 不应直接套用这些路径。

如果目标是生产性能，应优先使用 cuBLAS；如果目标是学习 GPU GEMM 优化，再继续改进 `cuda-tiled-32` 并与 cuBLAS 对比。

结果文件和图片已经由普通 Terminal 的 A100 benchmark 生成，并通过正确性校验；若需重新生成，必须在普通 Terminal 执行：

```bash
cd /home/hill/gemm
./gpu_run_benchmark.sh
```
