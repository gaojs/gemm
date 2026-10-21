# CUDA-GDB 使用方法

## 1. 工具简介

`cuda-gdb` 是 NVIDIA 提供的 CUDA 调试器，基于 GDB 扩展而来，用于调试包含 CPU 主机代码和 GPU kernel 的 CUDA 程序。

CUDA 程序通常包含两个执行空间：

- **Host（主机端）**：运行在 CPU 上，例如 `main`、普通 C/C++ 函数以及 CUDA Runtime API 调用。
- **Device（设备端）**：运行在 GPU 上，例如使用 `__global__`、`__device__` 声明的 CUDA 函数。

`cuda-gdb` 可以在同一个调试会话中管理 CPU 和 GPU，但两者的线程、寄存器、调用栈和单步语义仍然是分开的。

```text
CPU 执行 main()
    │
    │ 启动 CUDA kernel
    ▼
GPU 执行 kernel
    │
    │ kernel 完成
    ▼
CPU 继续执行同步、数据拷贝和结果校验
```

## 2. 编译调试版本

### 2.1 CPU/GPU 源码级调试

建议使用 `-g -G -O0` 编译。以本项目 GPU GEMM 为例：

```bash
cd /home/hill/gemm

nvcc -g -G -O0 -std=c++14 -arch=sm_89 \
  cuda/gpu_runner.cu -lcublas -lcudart \
  -o gpu_gemm_debug.x
```

参数说明：

- `-g`：生成 CPU 主机端调试信息。
- `-G`：生成 GPU 设备端调试信息，支持 kernel 内部源码级调试。
- `-O0`：关闭优化，便于断点和单步执行。
- `-arch=sm_89`：为计算能力 8.9 的 GPU 生成设备代码，应根据实际 GPU 架构调整。

`-G` 会显著降低 GPU kernel 的执行性能，并可能增加寄存器使用量。因此，该版本仅用于调试，不用于正式性能测试。

### 2.2 保留源码行信息的性能版本

如果只需要将 profiler 的指标映射到 CUDA 源码行，而不需要完整的 GPU 单步调试，可以使用：

```bash
mkdir -p tmp_cuda_lineinfo_build
for source in common gemm_cpu gemm_naive gemm_naive_ilp4 gemm_naive_1x2 \
  gemm_tiled_32 gemm_reg_2x2 gemm_reg_4x4 gemm_reg_4x4_64 \
  gemm_cublas gpu_runner; do
  nvcc -O3 -lineinfo -std=c++14 -arch=sm_89 -dc \
    -c "cuda/${source}.cu" -o "tmp_cuda_lineinfo_build/${source}.o"
done
nvcc -O3 -lineinfo -std=c++14 -arch=sm_89 \
  tmp_cuda_lineinfo_build/*.o -lcublas -lcudart \
  -o gpu_gemm_lineinfo.x
```

区别如下：

| 编译选项            | 用途                   |
| --------------- | -------------------- |
| `-g -G -O0`     | GPU 源码级断点、变量查看和单步调试  |
| `-g -lineinfo`  | CPU 调试信息及较轻量的源码行映射   |
| `-O3 -lineinfo` | 保持优化，主要服务于 Nsight 分析 |

## 3. 启动 cuda-gdb

直接调试：

```bash
cuda-gdb ./gpu_gemm_debug.x
```

也可以在命令行中预先执行调试命令：

```bash
cuda-gdb \
  -ex 'break main' \
  -ex 'run 1024' \
  ./gpu_gemm_debug.x
```

进入调试器后，可使用普通 GDB 命令：

```gdb
run 1024
continue
next
step
finish
backtrace
info locals
print variable_name
```

退出：

```gdb
quit
```

## 4. 调试 CPU 主机代码

CPU 端的普通函数和 CUDA Runtime API 调用，使用方式与 GDB 基本相同：

```gdb
break main
run 1024
next
step
continue
```

为函数设置断点：

```gdb
break run_gpu_version
break launch_naive
break gemm_naive_kernel
```

查看调用栈和局部变量：

```gdb
backtrace
info locals
info args
print m
print n
print k
```

查看 CPU 汇编：

```gdb
disassemble /m main
disassemble /m run_gpu_version
```

查看当前 CPU 指令和寄存器：

```gdb
x/i $pc
display/i $pc
info registers
```

注意：如果程序使用了较高优化级别，变量可能被优化掉，源码行也可能与实际执行顺序不完全一致。需要稳定单步时，优先使用 `-O0`。

## 5. 调试 GPU kernel

假设 CUDA 源码中有以下 kernel：

```cpp
__global__ void gemm_naive_kernel(
    int m, int n, int k,
    const double *a,
    const double *b,
    double *c) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m && col < n) {
    double sum = 0.0;
    for (int p = 0; p < k; ++p) {
      sum += a[row * k + p] * b[p * n + col];
    }
    c[row * n + col] = sum;
  }
}
```

设置 GPU kernel 断点：

```gdb
break gemm_naive_kernel
run 1024
```

也可以让程序在应用程序启动 kernel 时暂停：

```gdb
set cuda break_on_launch application
run 1024
```

进入 kernel 后，可以执行：

```gdb
next
step
print row
print col
print p
print sum
continue
```

查看 GPU 当前指令：

```gdb
x/i $pc
display/i $pc
```

如果调试器支持，可以查看 GPU 反汇编：

```gdb
disassemble
```

## 6. 查看和切换 CUDA 线程

GPU kernel 的执行组织为 grid、block、thread。调试时首先查看 CUDA 执行状态：

```gdb
info cuda kernels
info cuda threads
info cuda blocks
```

查看 CUDA 调试命令帮助：

```gdb
help cuda
help cuda thread
help cuda kernel
```

不同 CUDA-GDB 版本的线程选择命令可能略有差异。通常可以使用 CUDA-GDB 提供的线程选择命令切换当前 GPU 线程，然后再执行：

```gdb
print row
print col
x/i $pc
```

GPU 是 SIMT 架构。同一个 warp 中的线程通常执行同一条指令，因此 GPU 单步并不等同于 CPU 单线程单步，可能会影响当前 warp 中的相关线程。

## 7. CPU 与 GPU 的切换

`cuda-gdb` 可以在一个会话中先调试 CPU，再调试 GPU：

```gdb
break main
break gemm_naive_kernel
run 1024
```

程序首先在 CPU 的 `main` 处暂停。使用：

```gdb
continue
```

程序启动 kernel 后，会在 GPU kernel 断点处暂停。此时可以查看：

```gdb
info cuda kernels
info cuda threads
print row
print col
x/i $pc
```

GPU kernel 完成后继续运行：

```gdb
continue
```

程序回到 CPU 端后，可以继续使用：

```gdb
backtrace
info locals
disassemble /m
```

因此：

> `cuda-gdb` 是统一的调试入口，但 CPU 和 GPU 仍是两个独立的执行空间；不能把 CPU 指令和 GPU 指令当作一条连续的指令流单步执行。

## 8. 常用断点与执行命令

### 8.1 断点管理

```gdb
break main
break gemm_naive_kernel
info breakpoints
disable 1
enable 1
delete 1
clear main
```

### 8.2 执行控制

```gdb
run 1024
continue
next
step
finish
until <行号>
```

### 8.3 查看变量和内存

```gdb
print m
print row
print col
x/16f a
x/16gx a
```

### 8.4 查看线程和栈

```gdb
info threads
thread
thread <线程编号>
backtrace
```

CUDA 线程相关信息使用：

```gdb
info cuda threads
info cuda blocks
info cuda kernels
```

## 9. 查看完整/最终 ELF 中的 CUDA 代码

本文中的“完整 ELF”和“最终 ELF”指的是同一个文件：项目完整编译后生成的最终可执行文件 `gpu_gemm_profile.x`。它是 CPU 端 Linux ELF，同时在其中嵌入 CUDA fatbin；CPU 直接执行 ELF，CUDA Runtime 或驱动从其中加载 GPU 设备代码。

### 9.1 查看 ELF 主机部分

```bash
file gpu_gemm_profile.x
readelf -h gpu_gemm_profile.x
readelf -l gpu_gemm_profile.x
readelf -S gpu_gemm_profile.x
ldd gpu_gemm_profile.x
```

查看 CPU x86-64 汇编：

```bash
objdump -d -Mintel gpu_gemm_profile.x
objdump -S -Mintel gpu_gemm_debug.x
```

普通 `objdump` 主要用于 CPU ELF 部分，不能正确反汇编 CUDA fatbin 中的 GPU 指令。

### 9.2 使用 cuobjdump 提取 PTX 和 SASS

PTX 是 **Parallel Thread Execution** 的缩写，是 NVIDIA 定义的 GPU 中间指令表示。

SASS 是Streaming Assembly，是目标 GPU 实际执行的机器指令。

```text
CUDA C++ 源码 → PTX → SASS
```

从最终 ELF 中查看 PTX：

```bash
cuobjdump --dump-ptx gpu_gemm_profile.x
```

从最终 ELF 中查看 SASS：

```bash
cuobjdump --dump-sass gpu_gemm_profile.x
```

查看 CUDA 设备相关 ELF 信息：

```bash
cuobjdump --dump-elf gpu_gemm_profile.x
```

保存到文件：

```bash
cuobjdump --dump-ptx gpu_gemm_profile.x > gpu_gemm.ptx
cuobjdump --dump-sass gpu_gemm_profile.x > gpu_gemm.sass
cuobjdump --dump-elf gpu_gemm_profile.x > gpu_gemm.cuda-elf.txt
```

### 9.3 生成独立 kernel 的 PTX/CUBIN

各个 `gemm_*.cu` 文件只包含 kernel 和 launcher，版本注册语句统一放在 `gpu_runner.cu` 中。因此可以直接编译单个 `gemm_*.cu` 文件生成设备代码，不需要再额外移除 `GpuVersion` 注册对象。

要查看完整程序中实际嵌入的 PTX 或 SASS，推荐直接从最终 ELF 提取：

```bash
cuobjdump --dump-ptx gpu_gemm_profile.x > gpu_gemm.ptx
cuobjdump --dump-sass gpu_gemm_profile.x > gpu_gemm.sass
```

如果需要单独生成 `gemm_naive` 的 PTX/CUBIN，可以直接使用 `cuda/gemm_naive.cu`。临时输出文件使用 `tmp_` 前缀，便于与正式构建产物区分：

```bash
nvcc -arch=sm_89 -ptx \
  cuda/gemm_naive.cu \
  -o tmp_gemm_naive.ptx

nvcc -arch=sm_89 -cubin \
  cuda/gemm_naive.cu \
  -o tmp_gemm_naive.sm_89.cubin
```

这些 `tmp_` 文件只适合提取设备代码；正式构建和运行使用 `cuda/gpu_runner.cu`，它包含各版本实现。如果不再需要这些临时文件，可以删除：

```bash
rm -f tmp_gemm_naive.ptx tmp_gemm_naive.sm_89.cubin
```

查看指定 kernel：

```bash
grep -n -A80 -B5 gemm_naive_kernel gemm_naive.ptx
```

### 9.4 反汇编 CUBIN

```bash
nvdisasm gemm_naive.sm_89.cubin
```

需要源码行映射时：

```bash
nvdisasm -g gemm_naive.sm_89.cubin
```

`nvdisasm` 主要用于反汇编 CUBIN 中的 SASS；查看最终 ELF 中的 CUDA 代码时，优先使用 `cuobjdump`。

## 10. 工具职责总结

| 工具 | 主要职责 | 适用对象 |
|---|---|---|
| `cuda-gdb` | 运行时断点、单步、变量和线程调试 | CPU 主机代码与 GPU kernel |
| `gdb` | 普通 CPU 调试 | 不含 GPU 调试需求的 ELF 主机代码 |
| `readelf` | 查看 ELF 头、段、节和动态信息 | Linux 主机 ELF |
| `objdump` | 查看 CPU x86-64 反汇编和源码映射 | ELF 中的 CPU 代码 |
| `cuobjdump` | 从 ELF/fatbin 提取 PTX、SASS 和 CUDA ELF 信息 | 最终 CUDA 可执行文件或 fatbin |
| `nvdisasm` | 反汇编 GPU CUBIN 中的 SASS | 独立 CUBIN |
| `nvcc -ptx` | 从 CUDA 源码生成 PTX | CUDA 源文件或独立 kernel 源文件 |
| `nvcc -cubin` | 为指定 GPU 架构生成 CUBIN | CUDA 源文件或独立 kernel 源文件 |
| `nsys` | 查看整个程序时间线、CUDA API、kernel 和内存拷贝 | 系统级性能分析 |
| `ncu` | 分析单个 kernel 的计算、访存、occupancy 等指标 | kernel 级性能分析 |

推荐按以下顺序使用：

```text
cuda-gdb       → 调试运行过程
readelf/objdump → 查看 CPU ELF 和 CPU 汇编
cuobjdump      → 从最终 ELF 提取 CUDA 设备代码
nvdisasm       → 反汇编独立 CUBIN 的 GPU 指令
nsys/ncu       → 分析运行时间线和 GPU 性能瓶颈
```

## 11. 保留 nvcc 中间文件

使用 `--keep` 可以保留 nvcc 编译 `gpu_runner.cu` 时产生的中间文件：

```bash
mkdir -p build_keep
nvcc -g -G -O0 -std=c++14 -arch=sm_89 \
  --keep --keep-dir build_keep \
  cuda/gpu_runner.cu -lcublas -lcudart -o gpu_gemm_debug.x
```

可能出现的文件包括：

| 文件        | 含义                          |
| --------- | --------------------------- |
| `.ptx`    | GPU 中间表示                    |
| `.cubin`  | 针对特定 GPU 架构的设备二进制           |
| `.fatbin` | 多个 GPU 架构设备代码的集合            |
| `.o`      | 目标文件                        |
| `.x`      | 最终 Linux 主机 ELF，可能嵌入 fatbin |

## 12. 调试常见问题

### 12.1 找不到 cuda-gdb

```bash
which cuda-gdb
cuda-gdb --version
```

如果 CUDA 安装目录未加入 `PATH`，可临时设置：

```bash
export PATH=/usr/local/cuda/bin:$PATH
```

根据实际安装路径，也可能需要使用：

```bash
/usr/local/cuda-12.8/bin/cuda-gdb ./gpu_gemm_debug.x
```

### 12.2 GPU 断点不生效

优先确认：

1. 使用 `-g -G` 编译，而不是只有 `-lineinfo`。
2. 调试器和可执行文件来自同一份源码与构建结果。
3. kernel 确实被执行到。
4. `-arch` 与实际 GPU 架构兼容。
5. 没有使用会删除或重排大量代码的高等级优化。

### 12.3 程序运行很慢

GPU 调试版本通常会关闭优化，并且调试器会暂停 GPU 执行，这是正常现象。调试运行时间不能用于性能比较。

### 12.4 使用 cuBLAS 时无法进入 GEMM 内部

`cublasDgemm` 的内部 kernel 属于 NVIDIA 库代码，通常不能像本项目自己的 `gemm_naive_kernel` 一样进行源码级单步。对于 cuBLAS，使用 Nsight Systems 和 Nsight Compute 分析其实际启动的 GPU kernel 更合适。

### 12.5 调试后恢复性能测试

不要使用 `-G` 编译的程序做正式基准。性能测试应重新使用优化编译：

```bash
nvcc -O3 -lineinfo -std=c++14 -arch=sm_89 \
  cuda/gpu_runner.cu -lcublas -lcudart \
  -o gpu_gemm_profile.x
```

## 13. 常用命令速查

```bash
# 编译 GPU 调试版本
nvcc -g -G -O0 -std=c++14 -arch=sm_89 \
  cuda/gpu_runner.cu -lcublas -lcudart \
  -o gpu_gemm_debug.x

# 启动统一 CPU/GPU 调试器
cuda-gdb ./gpu_gemm_debug.x

# 从完整程序的 ELF 提取 PTX（推荐）
cuobjdump --dump-ptx gpu_gemm_profile.x > gpu_gemm.ptx

# 直接从 kernel 源文件生成 PTX
nvcc -arch=sm_89 -ptx \
  cuda/gemm_naive.cu -o tmp_gemm_naive.ptx

# 从最终 ELF 提取 SASS
cuobjdump --dump-sass gpu_gemm_profile.x > gpu_gemm.sass

# 生成 CUBIN 并反汇编
nvcc -arch=sm_89 -cubin \
  cuda/gemm_naive.cu -o tmp_gemm_naive.sm_89.cubin
nvdisasm -g tmp_gemm_naive.sm_89.cubin

# 查看 CPU ELF 汇编
objdump -d -Mintel gpu_gemm_profile.x
```