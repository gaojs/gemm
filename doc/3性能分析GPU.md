# GPU naive GEMM 性能分析

## 1. 分析对象与环境

分析对象为 `cuda/gemm_naive.cu` 中的 `gemm_naive_kernel`，计算双精度方阵 GEMM：`C = A × B`。线程块为 `16×16`，每个线程计算一个输出元素，并沿 K 维逐项累加。测试规模为 `N=1024`。

| 项目 | 环境/配置 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090 D，Compute Capability 8.9 |
| Nsight Compute | 2024.6.2 |
| CUDA 编译 | `nvcc -O3 -std=c++14 -arch=sm_89 -lineinfo` |
| 被分析 kernel | `gemm_naive_kernel` |
| N=1024 时的 grid/block | grid `(64,64,1)`，block `(16,16,1)` |

这里的 grid/block 配置只针对本文分析的 `cuda/gemm_naive.cu`，并不是所有版本都相同。各手写 kernel 的启动配置如下；当 `m=n=1024` 时，表中的整除结果分别为对应的实际 grid：

| 版本 | block | grid 计算方式 | 每线程主要计算范围 |
| --- | --- | --- | --- |
| `cuda-naive` | `(16,16,1)` | `((n+15)/16,(m+15)/16,1)` | 1 个输出元素 |
| `naive-ilp4` | `(16,16,1)` | `((n+15)/16,(m+15)/16,1)` | 1 个输出元素，4 条累加链 |
| `naive-1x2` | `(16,16,1)` | `((n+31)/32,(m+15)/16,1)` | 同一行相邻 2 个输出元素 |
| `tiled-32` | `(32,32,1)` | `((n+31)/32,(m+31)/32,1)` | 共享内存 32×32 tile |
| `reg-2x2` | `(16,16,1)` | `((n+31)/32,(m+31)/32,1)` | 2×2 个输出元素 |
| `reg-4x4` | `(8,8,1)` | `((n+31)/32,(m+31)/32,1)` | 4×4 个输出元素 |
| `reg-4x4-64` | `(16,16,1)` | `((n+63)/64,(m+63)/64,1)` | 4×4 个输出元素，覆盖 64×64 tile |
| `cublas-dgemm` | 由 cuBLAS 决定 | 由 cuBLAS 内部决定 | 不能用本项目 kernel 配置描述 |

## 2. 编译被分析程序

从项目根目录执行。`LD_LIBRARY_PATH` 中优先使用系统驱动库，避免加载到不兼容的 `libcuda.so.1`：

```bash
cd /home/hill/gemm
export LD_LIBRARY_PATH=/usr/lib64:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
nvcc -O3 -std=c++14 -arch=sm_89 -lineinfo \
  cuda/gpu_runner.cu -lcublas -lcudart -o gpu_gemm_profile.x
```

`-lineinfo` 用于保留源代码行映射，不需要像 `-G` 那样关闭设备优化。其他架构需把 `sm_89` 换成对应架构，例如 A100 使用 `sm_80`。

## 3. Nsight Systems：端到端活动时间线

### 采集与汇总

```bash
nsys profile --trace=cuda,cublas --sample=none --cpuctxsw=none \
  --force-overwrite=true --output=gemm_naive \
  ./gpu_gemm_profile.x 1024

nsys stats --force-export=true \
  --report cuda_api_sum \
  --report cuda_gpu_kern_sum \
  --report cuda_gpu_mem_time_sum \
  gemm_naive.nsys-rep

nsys stats --report cuda_gpu_trace gemm_naive.nsys-rep
```

Nsight Systems 2024.6 使用的 report 名为 `cuda_api_sum`、`cuda_gpu_kern_sum`、`cuda_gpu_mem_time_sum`、`cuda_gpu_trace`。较旧版本可能有不同命名，可先运行 `nsys stats --help-reports` 查询。报告可用 `nsys-ui ./gemm_naive.nsys-rep` 打开。输出文件保存在当前目录 `/home/hill/gemm`，便于与源码和分析结果一起管理。

### 实测摘要

本次 `cuda_api_sum` 是整个 runner 进程的汇总，包含 CPU、所有 GPU kernel、cuBLAS 及初始化，不是 naive 单版本的独立统计：

| API | 调用次数 | 总耗时 |
| --- | ---: | ---: |
| `cudaFree` | 148 | 140.070 ms |
| `cudaMemcpy` | 144 | 133.709 ms |
| `cudaDeviceSynchronize` | 100 | 99.895 ms |
| `cudaMalloc` | 147 | 18.050 ms |
| `cudaLaunchKernel` | 42 | 2.737 ms |

GPU kernel 汇总中 `gemm_naive_kernel` 共 6 次启动（预热 1 次、正式计时 5 次），总 GPU 时间 `11.646 ms`，平均约 `1.941 ms/次`。GPU 内存拷贝汇总是全程序范围：H2D 96 次、总 `56.698 ms`；D2H 48 次、总 `44.991 ms`。由于每个版本每轮都分配、复制并释放显存，端到端耗时不能简单等同于 kernel 时间；需要在时间线上按 kernel 名称及相邻 memcpy 活动识别目标启动。

### 判读方法

1. 在 `cuda_gpu_trace` 或 Nsight Systems 时间线上定位 `gemm_naive_kernel`，核对 N=1024 的 grid/block，并观察其前后的两次 H2D、D2H 和同步/分配释放调用。
2. 用 kernel 区间衡量设备计算时间，用 memcpy 区间查看传输开销；CUDA API 的主机耗时与 GPU 执行耗时是不同口径。
3. `cuda_api_sum`、kernel summary、memcpy summary 均覆盖整个 runner，不可将总数直接归因给 naive。隔离特定版本应在 GUI 时间线上核对顺序，或使用带 NVTX 范围的专用程序采集。
4. `cudaMalloc/cudaFree` 的主机 API 耗时不等于 GPU kernel 时间；结合 API trace 和 GPU 时间线分析其调用/同步开销。

## 4. Nsight Compute：naive kernel 计算特性

### 采集命令

runner 在每个 GPU 版本中依次执行一次预热和 5 次正式重复。下面只采集 `gemm_naive_kernel`，跳过其第一个匹配启动（预热），并采集之后的一个匹配启动：

```bash
ncu --set basic \
  --kernel-name 'regex:gemm_naive_kernel' \
  --launch-skip 1 --launch-count 1 \
  --export gemm_naive \
  ./gpu_gemm_profile.x 1024

ncu --import ./gemm_naive.ncu-rep
```

需要进一步确认计算管线时，可单独采集 `ComputeWorkloadAnalysis`：

```bash
ncu --section ComputeWorkloadAnalysis \
  --kernel-name 'regex:gemm_naive_kernel' \
  --launch-skip 1 --launch-count 1 \
  --export gemm_naive_compute \
  ./gpu_gemm_profile.x 1024

ncu --import ./gemm_naive_compute.ncu-rep
```

本次 basic 采样报告显示 kernel 为 `(64,64,1)×(16,16,1)`，与 N=1024 的配置一致。NCU 对同一次 kernel 可能进行多 pass/replay；报告值用于分析，不是正式性能计时。

### 实测指标与解释

| 指标 | 实测值 | 解读 |
| --- | ---: | --- |
| Compute (SM) Throughput | 96.30% | SM 计算管线利用率高 |
| DRAM Throughput | 0.80% | kernel 期间 DRAM 吞吐占峰值比例很低，不像是 DRAM 带宽瓶颈 |
| L2 Cache Throughput | 13.74% | L2 未接近饱和 |
| Memory Throughput | 24.11% | 综合内存吞吐不是主要饱和项 |
| Achieved Occupancy | 94.56% | 实际活跃 warp 占比较高 |
| Registers Per Thread | 40 | 寄存器使用量；报告的寄存器资源限制每 SM 可驻留 block 数为 6 |
| Duration | 2.15 ms | NCU 单次采样下的 kernel 时长，受 profiler 和 replay 影响 |
| FP64 pipeline utilization | 97.7% | ComputeWorkloadAnalysis 判定 FP64 是最高利用率的计算管线 |

### 分析结论

对该朴素 GEMM，在 RTX 4090 D、N=1024 的此配置下，NCU 指标显示主要限制来自 FP64 计算管线，而非显存带宽：SM/FP64 利用率高，DRAM 吞吐较低。占用率也较高，因此单纯提高 occupancy 的空间有限。可据此把后续优化重点放在提高每次加载数据的复用、减少重复访存/指令及改善计算吞吐；是否有效，必须通过同一设备上未挂 profiler 的正式基准验证。

## 5. 性能数字的使用边界

一次未挂 profiler 的 runner 测试中，N=1024 的 `cuda-naive` 端到端结果为约 `419.94 GFLOPS`，最大绝对误差为 `3.55e-14`。该计时包含显存分配、H2D、kernel、D2H 和释放；它不是 kernel-only GFLOPS。Nsight 运行期间程序输出的 GFLOPS 会受采集与 replay 显著干扰，不作为性能结论。Nsight Systems 的 kernel 汇总时间和 NCU 的单 kernel duration 也不能与端到端 GFLOPS 混为一谈。
