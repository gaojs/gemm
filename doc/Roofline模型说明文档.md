# Roofline模型说明文档

## 目录

1. [什么是Roofline模型](#1-什么是roofline模型)
2. [Roofline模型的组成部分](#2-roofline模型的组成部分)
3. [如何建立Roofline](#3-如何建立roofline)
4. [本CPU的Roofline计算](#4-本cpu的roofline计算)
5. [GEMM性能在Roofline上的位置](#5-gemm性能在roofline上的位置)
6. [工具方法](#6-工具方法)

---

## 1. 什么是Roofline模型

Roofline模型是一种可视化的性能分析工具，用于识别程序性能的瓶颈。它将程序的性能（GFLOPS）与计算强度（Computational Intensity，即每字节内存访问执行的浮点运算次数）关联起来，形成一个类似屋顶的图形。

**核心思想**：
- 程序性能受限于两个因素：**计算能力**（CPU峰值FLOPS）和**内存带宽**（内存访问速度）
- 当计算强度较低时，性能受限于内存带宽（内存墙）
- 当计算强度较高时，性能受限于计算能力（计算墙）

---

## 2. Roofline模型的组成部分

### 2.1 计算强度 (Computational Intensity)

计算强度定义为：每访问1字节内存所执行的浮点运算次数。

公式：
```
计算强度 (FLOPs/byte) = 总浮点运算次数 / 总内存访问字节数
```

对于GEMM (C = A × B + C)，矩阵大小为N×N：
- 浮点运算次数：2 × N³（每个元素需要N次乘法和N次加法）
- 内存访问字节数：约3 × N² × 8（读取A、B，写入C，每个元素8字节）
- 计算强度 ≈ (2 × N³) / (24 × N²) = N / 12

### 2.2 内存带宽线

内存带宽线表示在给定计算强度下，仅受内存带宽限制的最大性能。

公式：
```
性能 (GFLOPS) = 内存带宽 (GB/s) × 计算强度 (FLOPs/byte)
```

这是一条从原点出发的直线，斜率为内存带宽。

### 2.3 计算能力线

计算能力线表示CPU能够达到的峰值性能。

公式：
```
性能 (GFLOPS) = 峰值FLOPS
```

这是一条水平线，高度为CPU的峰值浮点运算能力。

### 2.4 拐点 (Knee Point)

内存带宽线和计算能力线的交点称为拐点。

公式：
```
拐点计算强度 = 峰值FLOPS / 内存带宽
```

- 当程序计算强度 < 拐点：性能受内存带宽限制
- 当程序计算强度 > 拐点：性能受计算能力限制

---

## 3. 如何建立Roofline

### 3.1 方法一：手动计算

步骤：
1. **测量内存带宽**：使用STREAM基准测试
2. **计算峰值FLOPS**：根据CPU架构和频率
3. **计算拐点**：峰值FLOPS / 内存带宽
4. **绘制图形**：以计算强度为横轴，性能为纵轴

### 3.2 方法二：使用LIKWID

LIKWID是一款轻量级性能监控工具，可以测量内存带宽和收集性能计数器数据。

**注意**：当前系统安装的LIKWID版本（5.1）不包含`likwid-roofline`命令（该命令从5.2版本开始提供）。可以使用以下方法获取性能数据：

安装：
```bash
sudo apt-get install likwid
```

使用：
```bash
# 测量内存带宽（在CPU socket 0上运行，使用1GB内存）
likwid-bench -t stream -w S0:1GB

# 测量内存带宽（单核心运行，使用1GB内存）
likwid-bench -t stream -w S0:1GB:1

# 使用perfctr收集性能计数器数据（-f强制覆盖被占用的寄存器）
likwid-perfctr -f -C 0 -g FLOPS_DP ./test_MMult.x
```

可用的LIKWID命令：
- `likwid-bench` - 微基准测试（包括STREAM）
- `likwid-perfctr` - 性能计数器监控
- `likwid-perfscope` - 性能范围可视化
- `likwid-topology` - CPU拓扑信息

---

## 4. 本CPU的Roofline计算

### 4.1 CPU信息

| 参数 | 值 |
|------|-----|
| CPU型号 | Intel(R) Xeon(R) Silver 4210 CPU @ 2.20GHz |
| 最大频率 | 3.2 GHz |
| 实际运行频率 | ~2.95 GHz |
| L3缓存 | 27.5 MiB |
| 核心数 | 20 |
| 架构 | Skylake-SP |

### 4.2 内存带宽

| 测试工具 | 线程数 | 内存带宽 |
|---------|-------|---------|
| 自定义STREAM | 1 | 14.31 GB/s |
| LIKWID stream | 1 | 12.85 GB/s |
| LIKWID stream | 10 | 50.16 GB/s |

**单核心内存带宽**: **12.8-14.3 GB/s**（取平均值约13.6 GB/s）

### 4.3 峰值FLOPS计算

**AVX2 (256位)**:
```
峰值FLOPS = 2 (FMA单元) × 2 (操作/指令) × 4 (向量长度) × 频率
          = 2 × 2 × 4 × 2.95 GHz
          = 47.2 GFLOPS
```

**AVX-512 (512位)**:
```
峰值FLOPS = 2 × 2 × 8 × 2.95 GHz
          = 94.4 GFLOPS
```

### 4.4 拐点计算

**AVX2拐点**:
```
拐点计算强度 = 47.2 GFLOPS / 14.31 GB/s = 3.3 FLOPs/byte
```

**AVX-512拐点**:
```
拐点计算强度 = 94.4 GFLOPS / 14.31 GB/s = 6.6 FLOPs/byte
```

### 4.5 Roofline参数汇总

| 参数 | AVX2 | AVX-512 |
|------|------|---------|
| 内存带宽 | 14.31 GB/s | 14.31 GB/s |
| 峰值FLOPS | 47.2 GFLOPS | 94.4 GFLOPS |
| 拐点计算强度 | 3.3 FLOPs/byte | 6.6 FLOPs/byte |

---

## 5. GEMM性能在Roofline上的位置

### 5.1 GEMM计算强度

对于N×N矩阵的GEMM：
```
计算强度 ≈ N / 12
```

| 矩阵大小N | 计算强度 (FLOPs/byte) |
|-----------|---------------------|
| 64 | 5.3 |
| 128 | 10.7 |
| 256 | 21.3 |
| 512 | 42.7 |
| 1024 | 85.3 |

### 5.2 当前性能分析

当前GEMM实现性能：约**33-35 GFLOPS**

**内存带宽限制下的理论性能**:
```
理论性能 = 14.31 GB/s × 计算强度
```

| 矩阵大小N | 计算强度 | 理论峰值 (内存带宽限制) | 实际性能 | 效率 |
|-----------|---------|----------------------|---------|------|
| 64 | 5.3 | 75.8 GFLOPS | 25 GFLOPS | 33% |
| 128 | 10.7 | 153.1 GFLOPS | 29 GFLOPS | 19% |
| 256 | 21.3 | 304.8 GFLOPS | 33 GFLOPS | 11% |
| 512 | 42.7 | 611.0 GFLOPS | 35 GFLOPS | 6% |

**关键发现**:
1. 当前GEMM性能远低于内存带宽限制的理论峰值
2. 实际性能约为33-35 GFLOPS，几乎与矩阵大小无关
3. 效率仅6%-33%，说明**不是简单的内存带宽限制**，而是缓存效率问题

### 5.2.1 根本原因分析

**为什么效率这么低？**

这是一个非常重要的问题。效率低（6%-33%）表明当前实现存在严重的**缓存层次效率问题**，而非单纯的内存带宽限制。

**问题本质**：当前实现没有充分利用CPU的多级缓存（L1→L2→L3），导致：

1. **数据复用率低**
   - GEMM的计算强度理论上随矩阵大小N线性增长（N/12）
   - 但实际性能几乎不变（33-35 GFLOPS）
   - 这说明数据没有在缓存中被多次复用，每次计算都要从内存读取

2. **分块策略不完善**
   - 当前使用4x8的内部分块，但缺乏外层的缓存分块
   - L1缓存大小约32KB，L2约1MB，L3约27MB
   - 需要将矩阵划分为适合各级缓存的大小，才能充分复用数据

3. **内存访问模式不佳**
   - 矩阵B的列访问导致不连续内存访问
   - 即使使用了PackMatrixB，外层循环仍然频繁访问内存

**真正的瓶颈**：不是内存带宽，而是**缓存层次的利用效率**。当前实现每次从内存读取数据后，只进行了少量计算就丢弃了，没有充分利用缓存的高速特性。

### 5.2.2 Perf实测结果

使用perf收集的实际性能计数器数据：

| 指标 | 值 | 命中率 |
|------|-----|--------|
| cycles | 924亿 | - |
| instructions | 1050亿 | - |
| IPC | 1.14 | - |
| L1-dcache-loads | 308亿 | - |
| L1-dcache-load-misses | 130亿 | **58%** (42% miss) |
| LLC-loads | 58亿 | - |
| LLC-load-misses | 1.6亿 | **97%** (2.75% miss) |

**关键发现**：
1. **L1缓存命中率仅58%**（理想应>95%），这是性能瓶颈的根本原因
2. L3缓存命中率97%（很好），说明数据最终能被缓存，但路径太长
3. IPC仅1.14（理想可达2-4），说明CPU流水线因等待数据而频繁停顿

问题链条：
```
L1 miss (42%) → L2访问 → L2 miss → L3访问 (97%命中)
     ↓
 性能严重下降
```

### 5.2.3 优化方向

要提升性能，需要实现**多级分块（Multi-level Tiling）**：

1. **寄存器分块**（当前已实现）：4x8分块，利用AVX256寄存器
2. **L1缓存分块**：将矩阵划分为适合L1缓存的大小（如64x64）
3. **L2缓存分块**：将矩阵划分为适合L2缓存的大小（如512x512）
4. **L3缓存分块**：将矩阵划分为适合L3缓存的大小（如2048x2048）

通过多级分块，数据可以在各级缓存中被多次复用，从而大大提高计算强度和实际性能。

### 5.3 Roofline位置图

```
性能 (GFLOPS)
     ^
 94.4|-------/----- AVX-512计算墙
     |      /
 47.2|-----/------- AVX2计算墙
     |    /|
 35  |---/ |   当前GEMM性能
     |  /  |
     | /   |
     |/    |
  0 -+-----+-------> 计算强度 (FLOPs/byte)
     0    3.3    6.6

     ^     ^      ^
   原点  AVX2拐点 AVX-512拐点
```

---

## 6. 工具方法

### 6.1 STREAM内存带宽测试

编译和运行：
```bash
cd /home/hill/gemm
gcc -O3 -march=skylake-avx512 -mfma stream.c -o stream.x
taskset -c 0 ./stream.x
```

### 6.2 峰值FLOPS测试

创建测试程序：
```c
#include <stdio.h>
#include <time.h>
#include <x86intrin.h>

#define N 4096
#define ITERATIONS 1000000

int main() {
    __m512d a = _mm512_set1_pd(1.0);
    __m512d b = _mm512_set1_pd(2.0);
    __m512d c = _mm512_set1_pd(0.0);
    
    clock_t start = clock();
    for (int i = 0; i < ITERATIONS; i++) {
        c = _mm512_fmadd_pd(a, b, c);
    }
    clock_t end = clock();
    
    double seconds = (double)(end - start) / CLOCKS_PER_SEC;
    double flops = (double)ITERATIONS * 8 * 2 / seconds; // 8 elements × 2 operations (mul+add)
    
    printf("AVX-512 Peak FLOPS: %.2f GFLOPS\n", flops / 1e9);
    return 0;
}
```

编译和运行：
```bash
gcc -O3 -march=skylake-avx512 -mfma peak_flops.c -o peak_flops.x
taskset -c 0 ./peak_flops.x
```

### 6.3 Perf性能计数器

使用perf获取详细的性能数据：
```bash
perf stat -e cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses taskset -c 0 ./test_MMult.x
```

---

## 附录：术语表

| 术语 | 英文 | 解释 |
|------|------|------|
| FLOPS | Floating Point Operations Per Second | 每秒浮点运算次数 |
| GFLOPS | Giga FLOPS | 每秒10亿次浮点运算 |
| TFLOPS | Tera FLOPS | 每秒1万亿次浮点运算 |
| FMA | Fused Multiply-Add | 融合乘加指令，一条指令完成乘法和加法 |
| SIMD | Single Instruction, Multiple Data | 单指令多数据，向量化并行 |
| 计算强度 | Computational Intensity | 每字节内存访问执行的浮点运算次数 |
| 拐点 | Knee Point | Roofline图中内存带宽线与计算能力线的交点 |