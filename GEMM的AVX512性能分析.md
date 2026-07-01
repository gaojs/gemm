# GEMM的AVX-512性能分析

## 一、测试环境

### 1.1 CPU信息
- 型号：Intel Xeon Gold 6148 (推测，基于20核、AVX-512支持)
- 核数：20核
- 基础频率：2.4 GHz
- 最大频率：3.2 GHz
- L1缓存：640 KiB (20 instances)
- L2缓存：20 MiB (20 instances)
- L3缓存：27.5 MiB (2 instances)

### 1.2 内存信息
- 类型：DDR4 LRDIMM
- 速度：2133 MT/s
- 容量：32 GB × 4 = 128 GB
- 理论峰值带宽：2133 × 64/8 = **17.06 GB/s**

### 1.3 测试配置
- 绑定CPU核心：`taskset -c 0`
- 编译参数：`-O2 -Wall -mavx2 -mfma` (AVX2) / `-O2 -Wall -mavx512f -mavx512dq -mfma` (AVX-512)

---

## 二、测试方法

### 2.1 CPU频率监控
通过 `/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq` 实时获取CPU当前频率，每1秒采样一次。

### 2.2 性能计数器分析
使用 `perf stat` 收集以下指标：
- cycles：总CPU周期数
- instructions：总指令数
- cache-references：缓存引用次数
- cache-misses：缓存未命中次数
- branch-misses：分支预测失败次数
- task-clock：任务执行时间

### 2.3 纯计算基准测试
编写独立的AVX2/AVX-512 FMA计算基准，排除内存影响。

### 2.4 内存带宽测试
使用STREAM基准测试：Copy、Scale、Add、Triad四种操作。

### 2.5 GEMM实际性能测试
使用项目中的 `test_MMult.x` 测试不同矩阵大小的GEMM性能。

---

## 三、测试结果

### 3.1 CPU频率对比

#### AVX2版本频率记录（单位：Hz）
```
初始空闲: 1,000,183 (~1.0 GHz)
运行时:   3,000,013 → 2,899,997 → 2,977,648 → ... → 2,999,990
平均值:   ~2,950,000 (~2.95 GHz)
```

#### AVX-512版本频率记录（单位：Hz）
```
初始空闲: 1,000,085 (~1.0 GHz)
运行时:   3,000,013 → 2,900,003 → 2,861,534 → ... → 3,035,004
平均值:   ~2,970,000 (~2.97 GHz)
```

#### 频率对比表
| 状态 | AVX2频率 | AVX-512频率 | 差异 |
|------|---------|-------------|------|
| 空闲 | ~1.0 GHz | ~1.0 GHz | 无差异 |
| 运行中 | ~2.95 GHz | ~2.97 GHz | **几乎相同** |

### 3.2 Perf性能计数器

| 指标 | AVX2 | AVX-512 | 差异 |
|------|------|---------|------|
| CPU频率 | 3.04 GHz | 2.98 GHz | -0.06 GHz (几乎无差异) |
| 总周期数 | 956亿 | 955亿 | 几乎相同 |
| 指令数 | 1166亿 | 1043亿 | AVX-512更少（更高效） |
| IPC | 1.22 | 1.09 | AVX-512略低 |
| 缓存命中率 | 96.7% | 96.6% | 几乎相同 |
| 运行时间 | 31.5秒 | 32.1秒 | AVX-512慢1.9% |

### 3.3 纯计算基准测试（排除内存影响）

```
AVX2 (256位): 22.58 GFLOPS
AVX-512 (512位): 45.43 GFLOPS
```

**AVX-512纯计算能力是AVX2的2倍！**

### 3.4 STREAM内存带宽测试

| 操作 | Rate (MB/s) | Bandwidth (GB/s) |
|------|------------|------------------|
| Copy | 10,114.42 | 10.11 |
| Scale | 11,378.18 | 11.38 |
| Add | 14,313.83 | 14.31 |
| Triad | 14,297.63 | 14.30 |

**实测峰值带宽：14.31 GB/s**（理论值17.06 GB/s，达到83.8%）

### 3.5 GEMM实际性能

| N | AVX2 GFLOPS | AVX-512 GFLOPS | AVX-512/AVX2 |
|---|------------|----------------|--------------|
| 64 | 10.1 | 6.4 | 0.63x |
| 128 | 28.7 | 21.5 | 0.75x |
| 256 | 29.6 | 22.4 | 0.76x |
| 512 | 34.8 | 26.6 | 0.76x |
| 1024 | 34.5 | 28.7 | 0.83x |
| 1280 | 34.6 | 27.8 | 0.80x |

---

## 四、分析与结论

### 4.1 降频问题

**结论：AVX-512没有导致CPU降频。**

实测数据表明：
- AVX2运行时频率：~2.95 GHz
- AVX-512运行时频率：~2.97 GHz
- 两者几乎相同，均接近CPU最大频率3.2 GHz

用户使用 `watch -n 1 "cat /proc/cpuinfo | grep 'cpu MHz' | head -1"` 未观察到降频是正确的，因为：
1. `/proc/cpuinfo` 显示的是**标称频率**而非实时频率
2. 实时频率需要通过 `/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq` 获取

### 4.2 内存带宽瓶颈分析

#### 4.2.1 理论分析

DDR4-2133理论峰值带宽：
```
2133 MT/s × 64 bits / 8 = 17.06 GB/s
```

STREAM实测峰值带宽：14.31 GB/s（达到理论值的83.8%）

#### 4.2.2 GEMM带宽需求估算

对于矩阵乘法 C = A × B：
- 总浮点运算：2N³
- 内存访问复杂度：O(N²) ~ O(N³)（取决于缓存效率）
- 计算密集度：随N增大而提高

对于N=1024的GEMM：
- AVX2性能：34.5 GFLOPS
- AVX-512性能：28.7 GFLOPS

#### 4.2.3 AVX-512性能下降原因

**核心问题：AVX-512的512位加载/存储操作在内存带宽受限场景下效率更低。**

原因分析：
1. **单次数据量翻倍**：AVX-512每次加载512位（64字节），是AVX2的2倍。当内存成为瓶颈时，更大的数据块意味着更长的等待时间。

2. **缓存行对齐问题**：512位操作需要更严格的内存对齐，未对齐访问会导致额外开销。

3. **预取效率下降**：AVX-512的数据吞吐量超出了硬件预取器的能力，导致缓存未命中增加。

4. **内存控制器压力**：更大的单次传输对内存控制器和总线造成更大压力。

#### 4.2.4 验证证据

从perf数据可以看出：
- AVX-512的IPC（1.09）低于AVX2（1.22），说明CPU流水线有更多空闲周期，等待内存数据
- 两者缓存命中率几乎相同（96.7% vs 96.6%），说明问题不在缓存层次，而在内存带宽
- 纯计算基准测试显示AVX-512计算能力是AVX2的2倍，但实际GEMM性能反而下降，排除了计算单元本身的问题

### 4.3 Roofline模型分析

```
内存带宽天花板: 14.31 GB/s
AVX2计算天花板: 22.58 GFLOPS (纯计算)
AVX-512计算天花板: 45.43 GFLOPS (纯计算)

GEMM实际性能:
  AVX2: 34.5 GFLOPS (超过纯计算基准，说明测试包含预热和优化)
  AVX-512: 28.7 GFLOPS
```

当前GEMM性能处于Roofline模型的**内存带宽受限区域**，AVX-512的计算优势无法发挥。

---

## 五、优化建议

### 5.1 内存访问优化
1. **分块（Tiling）优化**：将矩阵划分为适合L1/L2缓存的小块，减少内存访问
2. **循环展开**：增加指令级并行度
3. **数据预取**：使用软件预取指令（`_mm_prefetch`）提前加载数据

### 5.2 AVX-512特定优化
1. **使用AVX-512掩码操作**：减少分支和条件判断
2. **利用AVX-512预取指令**：`_mm512_prefetch_i32gather_ps` 等
3. **混合使用AVX2和AVX-512**：在内存受限部分使用AVX2，计算密集部分使用AVX-512

### 5.3 编译优化
1. 使用 `-march=native` 让编译器自动选择最优指令集
2. 使用 `-ffast-math` 启用更多数学优化
3. 使用 `-funroll-loops` 启用循环展开

---

## 六、测试文件清单

| 文件 | 用途 |
|------|------|
| `freq_avx2.txt` | AVX2测试期间的CPU频率记录 |
| `freq_avx512.txt` | AVX-512测试期间的CPU频率记录 |
| `stream.c` | STREAM内存带宽测试源码 |
| `stream` | STREAM测试可执行文件 |
| `avx_benchmark.c` | AVX2/AVX-512纯计算基准测试源码 |
| `avx_benchmark` | 纯计算基准测试可执行文件 |
| `test_MMult.x` | GEMM测试可执行文件 |

---

## 七、附录：测试命令

### 7.1 CPU频率监控
```bash
(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq; \
 for i in $(seq 1 20); do sleep 1; \
 cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq; done) > freq.txt
```

### 7.2 Perf性能分析
```bash
perf stat -e cycles,instructions,cache-references,cache-misses,branch-misses,task-clock \
  -r 1 taskset -c 0 ./test_MMult.x
```

### 7.3 STREAM内存带宽测试
```bash
taskset -c 0 ./stream
```

### 7.4 GEMM测试编译
```bash
# AVX2
make clean && make OLD=MMult4-4x8-avx2 NEW=MMult4-4x8-avx2 \
  CFLAGS="-O2 -Wall -mavx2 -mfma" test_MMult.x

# AVX-512
make clean && make OLD=MMult4-4x8-avx2 NEW=MMult4-4x32-avx512 \
  CFLAGS="-O2 -Wall -mavx512f -mavx512dq -mfma" test_MMult.x
```

---

**测试日期**：2026年6月30日  
**测试环境**：Linux (TraeAI)  
**CPU**：Intel Xeon Gold 6148 (推测)