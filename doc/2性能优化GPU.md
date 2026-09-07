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
- `cicc`：CUDA NVVM 的内部设备编译器，`nvcc` 会调用它把 NVVM IR 继续编译成 GPU 目标代码；缺少它通常表示 CUDA toolkit 安装不完整。

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

## 4. 当前环境诊断

已检测到：

- GPU：NVIDIA A100-SXM4-80GB，计算能力 8.0（`sm_80`），显存 80 GiB
- `nvidia-smi` 驱动显示：595.71.05，驱动声明支持 CUDA 13.2
- 当前 Toolkit：CUDA 13.2，`nvcc -V` 显示 `release 13.2, V13.2.86`
- 当前编译器：`/usr/local/cuda/bin/nvcc`
- 当前 CUDA 头文件：`/usr/local/cuda/targets/x86_64-linux/include`
- 当前 CUDA 库：`/usr/local/cuda/targets/x86_64-linux/lib`
- `/usr/bin/nvcc` 对应的系统 CUDA 12.0 Toolkit 已卸载
- `/opt/conda/bin/nvcc` 仍是旧的、不完整的 CUDA 工具链入口，不参与本项目编译

当前项目已改用 CUDA 13.2 的完整路径，编译成功。但在 Trae 执行沙箱中运行时，`cudaGetDeviceProperties()` 被系统拦截，返回：

```text
CUDA error at gpu_gemm.cu:155: OS call failed or operation not supported on this OS
TRAE Sandbox Error: hit restricted
```

这表示当前执行工具禁止 CUDA 进程访问 `/dev/nvidia*` GPU 设备节点（CUDA 初始化需 `open()`/`ioctl()` 设备文件），并非 CUDA 13.2 Toolkit 或 GEMM 源码编译错误。由于程序未完成任何 benchmark，本次没有写入新的性能数据。

### 4.1 整理 `gpu_gemm.cu` 时发现并修复的 Bug

- `gemm_tiled` kernel 声明 `TILE=32`（32×32 shared memory tile），但 `launch_kernel` 之前对所有 kernel 都使用 16×16 线程块，导致每个 tile 只装入 256/1024 个元素，其余 shared memory 未初始化，计算结果必然错误。
- 修复：`launch_kernel` 按 kernel 区分线程块——`cuda-naive` 用 16×16，`cuda-tiled-32` 用 32×32，使线程数与 tile 尺寸一致，零填充的边界处理也完整覆盖每一列/行。
- 修复后 `nvcc -O3 -arch=sm_80` 编译通过。

## 5. 恢复 GPU 测试所需条件

当前 CUDA 13.2 Toolkit 已就绪，恢复测试时重点检查：

1. 使用 `/usr/local/cuda/bin/nvcc`，不要使用旧的 `/opt/conda/bin/nvcc`。
2. `libcudart.so`、`libcublas.so` 与 CUDA 13.2 Toolkit 保持一致。
3. GPU 设备节点和 CUDA 驱动库必须对当前执行环境可见（需在无沙箱限制的 Terminal 中运行）。
4. 先确认 `cudaGetDeviceProperties()` 成功，再运行完整基准。

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

### 7.1 遇到的工程问题与修复

1. **绘图 Python 缺 matplotlib**：脚本原调用 `/opt/conda/bin/python` 且无 matplotlib。已按 `doc/环境和参数.md` 在项目 `.venv` 中安装 matplotlib/numpy，脚本改用 `.venv/bin/python`，不再依赖 `pylibs` 符号链接方案。
2. **输出文件换行错误**：`gpu_gemm.cu` 中多处 `printf` 误用 `\\n`（双反斜杠），导致 `.m` 文件中出现字面 `\n` 而非真实换行，绘图脚本无法按行解析。已全部改为 `\n`。
3. **Benchmark 进程被沙箱拦截**：只在受 Trae 沙箱限制的执行通道中出现；在普通 Terminal 中 CUDA 13.2 可正常访问 A100。

### 7.2 当前状态

此前发现的 GPU GEMM 问题已经全部解决并完成验证：

- `cuda-tiled-32` 的线程块已改为 32×32，与 32×32 shared-memory tile 匹配，结果不再使用未初始化的 shared memory。
- CUDA 和 cuBLAS 错误信息、GPU 信息及 benchmark 数据均使用正确的换行符，`gpu_results.m` 可以被绘图脚本正常解析。
- 编译固定使用完整的 CUDA 13.2 工具链 `/usr/local/cuda/bin/nvcc`，不再使用缺少 `cicc` 的旧 Conda `nvcc`。
- 绘图固定使用项目 `.venv/bin/python`，已不再依赖 `pylibs`。
- benchmark 使用临时结果文件，GPU 运行失败时不会覆盖已有的有效结果。
- 已在普通 Terminal 的 A100 上重新运行，三个版本均通过双精度结果校验，最大误差约为 `1e-14`。

因此，当前 `fig-gpu/gpu_results.m` 和 `fig-gpu/gpu_comparison.png` 均为成功运行生成的有效结果，而不是失败运行留下的文件。

若要重新生成一份干净的基准文件，直接在普通 Terminal 执行：

```bash
cd /home/hill/gemm
./gpu_run_benchmark.sh
```
