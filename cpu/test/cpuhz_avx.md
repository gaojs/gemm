# CPU 频率测试方法与结果

---

## 一、测试环境

| 项目 | 说明 |
|------|------|
| **平台** | Linux (Ubuntu) |
| **CPU** | Intel(R) Xeon(R) Silver 4210 CPU @ 2.20GHz |
| **核心数** | 10 物理核心 / 20 逻辑核心 |
| **缓存** | 14080 KB L3 Cache |
| **架构** | x86-64 |
| **SIMD支持** | SSE, SSE2, AVX, AVX2, AVX-512 |

---

## 二、测试方法

### 2.1 使用 cpuhz.cpp 工具

**文件位置**: [cpuhz.cpp](file:///home/hill/gemm/test/cpuhz.cpp)

**编译命令**:
```bash
g++ -std=c++11 -O2 cpuhz.cpp -o cpuhz
```

**运行命令**:
```bash
./cpuhz
```

**测试原理**:
- 使用 RDTSC 指令（`__rdtsc()`）读取 CPU 时间戳计数器
- 通过高精度时钟（`std::chrono::high_resolution_clock`）测量实际时间
- 计算公式：`频率(GHz) = TSC周期数 / 时间(纳秒)`

**测试项目**:
1. **空闲频率测试**: 在 sleep 期间测量 CPU 频率
2. **满载频率测试**: 通过浮点运算（sin/cos）压测 CPU，触发睿频
3. **倍频计算**: 假设总线频率为 100 MHz，计算倍频

### 2.2 读取 sysfs 接口

**最大频率**:
```bash
cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq
# 输出: 3200000 (kHz)
```

**当前频率**:
```bash
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq
# 输出: 3200000 (kHz)
```

**Intel pstate 状态**:
```bash
cat /sys/devices/system/cpu/intel_pstate/max_perf_pct
# 输出: 100 (表示启用全部睿频)
```

### 2.3 读取 /proc/cpuinfo

```bash
cat /proc/cpuinfo | grep "model name" | head -1
# 输出: model name : Intel(R) Xeon(R) Silver 4210 CPU @ 2.20GHz

cat /proc/cpuinfo | grep "cpu MHz" | head -1
# 输出: cpu MHz     : 999.681 (当前运行频率，受电源管理影响)
```

---

## 三、测试结果

### 3.1 cpuhz.cpp 测试输出

```
================================================
          CPU 频率测试工具
================================================

=== CPU 信息 ===
model name      : Intel(R) Xeon(R) Silver 4210 CPU @ 2.20GHz
cpu MHz         : 999.681
cache size      : 14080 KB
siblings        : 10
cpu cores       : 10

=== 系统时钟信息 ===
硬件最大频率: 3200 GHz
硬件最小频率: 1000 GHz

=== 测量空闲频率 ===
迭代   0: 2.199 GHz
迭代  10: 2.195 GHz
迭代  20: 2.195 GHz
...
迭代  90: 2.195 GHz

=== 测量满载频率（睿频）===
满载频率: 2.195 GHz
迭代次数: 45497053
TSC周期数: 2194842240
耗时: 1000.000 ms

=== 测量倍频 ===
最小频率: 2.195 GHz
最大频率: 2.195 GHz
平均频率: 2.195 GHz

假设总线频率: 0.100 GHz
最小倍频: 21.9x
最大倍频: 21.9x
平均倍频: 21.9x
```

### 3.2 结果分析

| 参数 | 测量值 | 说明 |
|------|--------|------|
| **基础频率** | 2.2 GHz | CPU标称基础频率 |
| **空闲频率** | ~2.2 GHz | TSC频率，恒定不变 |
| **满载频率** | ~2.2 GHz | TSC频率（恒定），非实际运行频率 |
| **硬件最大频率** | 3.2 GHz | 通过 sysfs 接口获取的最大睿频 |
| **最大倍频** | 32x | 3.2 GHz / 0.1 GHz |

### 3.3 关键发现

1. **TSC频率恒定**: 该CPU支持 `constant_tsc` 标志，TSC频率始终保持为基础频率（2.2 GHz），不随实际运行频率变化
2. **睿频支持**: Intel pstate 驱动启用，`max_perf_pct = 100`，表示支持完整睿频
3. **实际频率**: TSC无法反映实际运行频率，需通过 sysfs 接口或性能测试推断
4. **理论峰值计算**:
   - 单核心: `8 FLOPs/cycle × 3.2 GHz = 25.6 GFLOPS`
   - 10核心: `8 × 10 × 3.2 = 256 GFLOPS`

---

## 四、GEMM 性能测试验证

### 4.1 测试结果

| 版本 | 峰值性能 (GFLOPS) | 达到理论峰值比例 |
|------|-------------------|------------------|
| MMult0 (基线) | ~1.5 | ~6% |
| MMult1 | ~3.5 | ~14% |
| MMult2_1x4 | ~5.5 | ~21% |
| MMult3_4x4 | ~12-18 | ~47-70% |
| MMult4-4x4-avx2 | ~21.5 | ~84% |

### 4.2 结论

- **MMult4-4x4-avx2** 达到最高性能 **21.5 GFLOPS**，约为理论峰值的 **84%**
- 性能提升路径：MMult0 → MMult1 → MMult2 → MMult3 → MMult4，逐步优化
- 从 MMult3 到 MMult4-4x4-avx2 的提升主要来自 AVX2 指令集的使用

---

## 五、参考资料

1. [Intel ARK - Xeon Silver 4210](https://ark.intel.com/content/www/us/en/ark/products/120508/intel-xeon-silver-4210-processor-14m-cache-2-20-ghz.html)
2. [RDTSC Instruction](https://en.wikipedia.org/wiki/Time_Stamp_Counter)
3. [Linux cpufreq 子系统](https://www.kernel.org/doc/html/latest/admin-guide/pm/cpufreq.html)
4. [Intel Turbo Boost Technology](https://www.intel.com/content/www/us/en/architecture-and-technology/turbo-boost/turbo-boost-technology.html)
