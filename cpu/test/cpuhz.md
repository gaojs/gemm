# CPU 主频、倍频和睿频原理与测试

---

## 一、基本概念

### 1.1 CPU 主频

**定义**：CPU 的核心时钟频率，即 CPU 每秒完成的时钟周期数。

**单位**：Hz（赫兹）、kHz（千赫兹）、MHz（兆赫兹）、GHz（吉赫兹）

**计算公式**：
```
主频 = 倍频 × 总线频率（外频）
```

### 1.2 倍频

**定义**：CPU 核心频率与总线频率的比值。

**作用**：通过调整倍频可以在不改变外部总线频率的情况下调整 CPU 频率。

**示例**：
- 总线频率：100 MHz
- 倍频：26x
- 主频 = 100 MHz × 26 = 2.6 GHz

### 1.3 睿频（Turbo Boost）

**定义**：Intel 的动态频率调整技术，允许 CPU 在满足功耗和温度条件时自动提高频率。

**特点**：
- 单核睿频：仅一个核心运行时可达到最高频率
- 多核睿频：多个核心同时运行时频率会降低
- 温度限制：温度过高时自动降频
- 功耗限制：功耗超过 TDP 时自动降频

**示例（Intel i5-13420H）**：
- 基础频率：2.6 GHz（倍频 26x）
- P核最大睿频：4.6 GHz（倍频 46x）
- E核最大睿频：3.4 GHz（倍频 34x）

---

## 二、工作原理

### 2.1 时钟信号生成

```
晶振 → 锁相环(PLL) → 总线频率 → 倍频器 → CPU主频
  100 MHz              100 MHz       26x      2.6 GHz
```

### 2.2 倍频调整机制

| 状态 | 倍频 | 频率 |
|------|------|------|
| 空闲 | 8x | 800 MHz |
| 轻度负载 | 15x | 1.5 GHz |
| 中等负载 | 26x | 2.6 GHz |
| 睿频（单核） | 46x | 4.6 GHz |
| 睿频（多核） | 30-40x | 3.0-4.0 GHz |

### 2.3 睿频触发条件

1. **功耗余量**：当前功耗 < TDP（45W）
2. **温度余量**：当前温度 < TJ Max（100°C）
3. **核心数量**：激活的核心数量越少，睿频越高

---

## 三、测试方法

### 3.1 测试工具：cpuhz.cpp

**文件位置**：[cpuhz.cpp](file:///home/hill/gemm/cpuhz.cpp)

**编译命令**：
```bash
g++ -std=c++11 -O2 cpuhz.cpp -o cpuhz
```

**运行命令**：
```bash
./cpuhz
```

### 3.2 测试原理

使用 **RDTSC 指令**（Read Time-Stamp Counter）测量 CPU 时钟周期数：

```c
uint64_t rdtsc() {
    return __rdtsc();  // 返回从开机以来的 CPU 时钟周期数
}
```

**频率计算公式**：
```
频率(GHz) = TSC周期数 / 时间(纳秒)
```

### 3.3 测试项目

| 测试项 | 方法 | 说明 |
|--------|------|------|
| 空闲频率 | sleep 期间测量 | CPU 处于低负载状态 |
| 满载频率 | 浮点运算压测 | CPU 处于高负载状态，触发睿频 |
| 倍频计算 | 频率 / 总线频率 | 假设总线频率为 100 MHz |

### 3.4 运行结果示例

```
=== 测量空闲频率 ===
迭代   0: 2.615 GHz
迭代  10: 2.611 GHz
...

=== 测量满载频率（睿频）===
满载频率: 2.611 GHz
迭代次数: 17616421
TSC周期数: 2611211096
耗时: 1000.000 ms

=== 测量倍频 ===
最小频率: 2.611 GHz
最大频率: 2.611 GHz
平均频率: 2.611 GHz

假设总线频率: 0.100 GHz
最小倍频: 26.1x
最大倍频: 26.1x
平均倍频: 26.1x
```

---

## 四、WSL2 环境限制

### 4.1 频率虚拟化

WSL2 运行在 Hyper-V 虚拟机中，CPU 频率被虚拟化：

| 限制 | 说明 |
|------|------|
| 频率报告 | 显示基础频率（2.6 GHz） |
| 睿频支持 | 虚拟化层可能不传递睿频信息 |
| cpufreq 接口 | 大部分不可用 |
| 真实频率 | 实际运行频率由 Windows 主机控制 |

### 4.2 在 Windows 中确认睿频

**方法1：任务管理器**
1. 打开任务管理器（Ctrl + Shift + Esc）
2. 切换到"性能"标签页
3. 选择"CPU"
4. 查看"最大频率"字段（应显示 4.6 GHz）

**方法2：CPU-Z**
1. 下载 CPU-Z：https://www.cpuid.com/softwares/cpu-z.html
2. 查看"Specification"标签页
3. 查看"Max Turbo"字段

**方法3：PowerShell**
```powershell
Get-CimInstance Win32_Processor | Select-Object Name, MaxClockSpeed
```

---

## 五、常见问题

### 5.1 为什么满载频率没有达到睿频？

**可能原因**：
1. **WSL2 限制**：虚拟化层不支持睿频报告
2. **温度过高**：CPU 温度超过 TJ Max
3. **功耗限制**：功耗超过 TDP
4. **多核运行**：多个核心同时运行时睿频降低

### 5.2 如何确认 CPU 是否支持睿频？

#### 方法1：查看 CPU 标志

```bash
# 查看 RDTSC 支持（返回 tsc 表示支持）
grep "flags" /proc/cpuinfo | head -1 | grep -o "tsc"
# 输出：tsc

# 查看所有与时间相关的标志
lscpu | grep tsc
# 输出示例：
# Flags: ... constant_tsc ... nonstop_tsc ... rdtscp ...
```

**TSC 相关标志含义**：
| 标志 | 含义 |
|------|------|
| `tsc` | 支持 RDTSC 指令 |
| `constant_tsc` | TSC 频率恒定，不受 CPU 频率变化影响 |
| `nonstop_tsc` | TSC 在 C 状态下继续运行 |
| `rdtscp` | 支持带处理器 ID 的 RDTSCP 指令 |

#### 方法2：查看 Intel pstate 驱动

```bash
cat /sys/devices/system/cpu/intel_pstate/status 2>/dev/null || echo "不支持 intel_pstate"
# WSL2 环境输出：不支持 intel_pstate
```

> **说明**：WSL2 环境下不支持 intel_pstate 驱动，因为虚拟化层不暴露该接口。

#### 方法3：在 Windows 中查看

```powershell
Get-CimInstance Win32_Processor | Select-Object Name, MaxClockSpeed
# 输出：
# Name                                 MaxClockSpeed
# 13th Gen Intel(R) Core(TM) i5-13420H          2100
```

> **说明**：`MaxClockSpeed` 返回的是 **P 核的基础频率（2.1 GHz）**，不是最大睿频（4.6 GHz）。要查看真正的睿频，请使用任务管理器或 CPU-Z。

### 5.3 当前环境测试结果解读

```bash
# 1. RDTSC 支持
grep "flags" /proc/cpuinfo | head -1 | grep -o "tsc"
# 输出：tsc（重复多次，表示每个逻辑 CPU 都支持）

# 2. TSC 标志详情
lscpu | grep tsc
# 输出：constant_tsc nonstop_tsc rdtscp tsc_reliable tsc_known_freq

# 3. Intel pstate 驱动
cat /sys/devices/system/cpu/intel_pstate/status
# 输出：不支持 intel_pstate（WSL2 限制）

# 4. Windows 报告的频率
Get-CimInstance Win32_Processor | Select-Object Name, MaxClockSpeed
# 输出：MaxClockSpeed = 2100（P核基础频率，非睿频）
```

**解读**：
1. **TSC 支持良好**：CPU 支持 RDTSC、constant_tsc、nonstop_tsc 等高级特性，频率测量结果可靠
2. **WSL2 限制**：不支持 intel_pstate 驱动，无法通过标准 Linux 接口控制睿频
3. **频率报告**：Windows 的 `MaxClockSpeed` 只报告基础频率（2.1 GHz），不反映睿频（4.6 GHz）

### 5.4 如何启用睿频？

**BIOS 设置**：
1. 重启电脑，进入 BIOS
2. 找到"Intel Turbo Boost Technology"选项
3. 设置为"Enabled"
4. 保存并退出

**Linux 内核参数（原生 Linux，非 WSL2）**：
```bash
# 查看当前睿频状态
cat /sys/devices/system/cpu/intel_pstate/max_perf_pct

# 设置最大性能百分比（100% 表示启用全部睿频）
echo 100 | sudo tee /sys/devices/system/cpu/intel_pstate/max_perf_pct
```

> **WSL2 用户注意**：intel_pstate 接口在 WSL2 中不可用，睿频由 Windows 主机管理。

### 5.5 intel_pstate 是什么？

**定义**：`intel_pstate` 是 Linux 内核的 Intel CPU 频率缩放驱动，用于替代旧的 `acpi-cpufreq` 驱动。

**主要作用**：
1. **CPU 频率调节**：根据负载动态调整 CPU 频率
2. **睿频控制**：启用或限制 Intel Turbo Boost
3. **性能模式选择**：在性能模式和节能模式之间切换
4. **HWP 协调**：与 Intel Hardware P-states (HWP) 特性协同工作

**主要接口**：
```
/sys/devices/system/cpu/intel_pstate/
├── max_perf_pct      # 最大性能百分比（0-100）
├── min_perf_pct      # 最小性能百分比（0-100）
├── no_turbo          # 是否禁用睿频（0=启用，1=禁用）
├── status            # 驱动状态（active/passive/off）
├── turbo_pct         # 睿频提升百分比
└── num_pstates       # P-state 数量
```

**与 acpi-cpufreq 的区别**：
| 特性 | intel_pstate | acpi-cpufreq |
|------|--------------|--------------|
| 适用 CPU | Intel 现代 CPU | 通用 CPU |
| 频率精度 | 高（精确到 P-state） | 中（受 ACPI 限制） |
| 睿频支持 | 完整支持 | 部分支持 |
| 性能优化 | 针对 Intel 优化 | 通用 |
| HWP 支持 | 支持 | 不支持 |

**WSL2 状态**：不支持该驱动，因为 Hyper-V 虚拟化层不暴露 P-state 接口。

### 5.6 WSL 是否支持睿频？

**结论**：WSL2 **不直接支持睿频控制**，但**实际运行时会利用睿频**。

**具体表现**：
- **频率报告**：WSL2 中 `lscpu` 报告的是基础频率（2.6 GHz）
- **实际性能**：实际 CPU 运行频率可能高于报告频率
- **睿频触发**：当 WSL 中的程序需要高性能时，Windows 主机仍会触发睿频
- **频率调节**：由 Windows 主机管理，WSL2 无法直接控制

**验证方法**：
```bash
# 方法1：使用 cpuhz 工具测试（结果可能显示 2.6 GHz）
./cpuhz

# 方法2：在 Windows 任务管理器中观察
# 打开任务管理器 → 性能 → CPU，查看"当前速度"是否会升高

# 方法3：使用 perf（原生 Linux）观察
# perf stat -e cycles ./cpuhz
```

### 5.7 WSL 的最大主频是多少？

**理论上**：WSL 中可以运行的程序性能上限是 **CPU 的最大睿频（4.6 GHz）**。

**实际上**：
- WSL2 虚拟化层**不会人为限制 CPU 性能**
- 实际运行频率由 Windows 主机根据负载动态调整
- 满载时可能达到 4.6 GHz（睿频）或更高
- 空闲时会降低到 800 MHz 甚至更低

**影响 WSL 性能的因素**：
| 因素 | 说明 |
|------|------|
| 虚拟化开销 | 约 5-10% 的性能损失 |
| 内存访问 | 跨虚拟化边界的内存访问较慢 |
| I/O 性能 | 文件系统虚拟化会降低 I/O 性能 |
| 睿频支持 | 正常，支持睿频 |
| 多核调度 | 正常，支持多核 |

**建议**：
- 对于 GEMM 等计算密集型任务，WSL2 的性能损失较小（约 5-10%）
- 对于 I/O 密集型任务，WSL2 的性能损失较大（可能 20-50%）
- 关键性能测试建议在原生 Linux 环境中进行

### 5.8 WSL2 环境实测结果解读

**测试环境**：
- **平台**：Windows 11 + WSL2 (Ubuntu 24.04)
- **CPU**：13th Gen Intel(R) Core(TM) i5-13420H
- **内核**：6.6.87.2-microsoft

**完整测试输出**：

```
================================================
          CPU 频率测试工具
================================================

=== CPU 信息 ===
model name      : 13th Gen Intel(R) Core(TM) i5-13420H
cpu MHz         : 2611.210
cache size      : 12288 KB
siblings        : 12
cpu cores       : 6
（重复 12 次，每个逻辑 CPU 一次）

=== 系统时钟信息 ===
（空 - WSL2 不支持 cpufreq 接口）

=== 测量空闲频率 ===
迭代   0: 2.619 GHz
迭代  10: 2.611 GHz
迭代  20: 2.611 GHz
迭代  30: 2.611 GHz
迭代  40: 2.611 GHz
迭代  50: 2.611 GHz
迭代  60: 2.611 GHz
迭代  70: 2.611 GHz
迭代  80: 2.611 GHz
迭代  90: 2.611 GHz

=== 测量满载频率（睿频）===
满载频率: 2.611 GHz
迭代次数: 26874977
TSC周期数: 2611211324
耗时: 1000.000 ms

=== 测量倍频 ===
最小频率: 2.611 GHz
最大频率: 2.611 GHz
平均频率: 2.611 GHz

假设总线频率: 0.100 GHz
最小倍频: 26.1x
最大倍频: 26.1x
平均倍频: 26.1x
```

**结果解读**：

#### 1. CPU 信息部分

| 字段 | 值 | 解读 |
|------|------|------|
| `model name` | 13th Gen Intel Core i5-13420H | 第13代 Intel i5-13420H |
| `cpu MHz` | 2611.210 | 基础频率 2.6 GHz |
| `cache size` | 12288 KB | 12MB L3 缓存 |
| `siblings` | 12 | 12 个逻辑 CPU（超线程） |
| `cpu cores` | 6 | 6 个物理核心（4P + 2E） |

> **注意**：每个逻辑 CPU 都会显示一次，导致 `model name` 等字段重复 12 次。

#### 2. 空闲频率部分

```
迭代 0: 2.619 GHz  （首次迭代略高，初始化开销）
迭代 10-90: 2.611 GHz（稳定在 2.6 GHz）
```

**解读**：
- CPU 在空闲时运行在 **2.6 GHz**（基础频率）
- WSL2 报告的是基础频率，不是空闲时的实际频率
- 首次迭代 2.619 GHz 略高是 TSC 读取的初始化开销

#### 3. 满载频率部分

```
满载频率: 2.611 GHz
迭代次数: 26874977（2687万次浮点运算）
TSC周期数: 2611211324（约 26.1 亿个周期）
耗时: 1000.000 ms
```

**计算验证**：
```
频率 = TSC周期数 / 耗时 = 2611211324 / 1秒 = 2.611 GHz
```

**解读**：
- 即使在满载情况下，WSL2 报告的频率仍是 **2.6 GHz**
- 这不代表 CPU 没有睿频，只是 WSL2 不报告实际睿频频率
- 实际 CPU 运行频率可能高于报告值
- 2687万次 sin/cos 运算在 1秒内完成，计算能力正常

#### 4. 倍频计算部分

```
倍频 = 频率 / 总线频率 = 2.611 / 0.1 = 26.1x
```

**解读**：
- 26.1x 是基础频率对应的倍频
- 实际睿频时倍频可达 46x（4.6 GHz）
- WSL2 中只能观察到基础倍频

#### 5. WSL2 限制总结

| 项目 | WSL2 实测 | 实际情况 | 说明 |
|------|----------|---------|------|
| 空闲频率 | 2.6 GHz | 800 MHz ~ 2.6 GHz | WSL2 报告上限 |
| 满载频率 | 2.6 GHz | 4.6 GHz（睿频） | WSL2 不报告睿频 |
| 倍频 | 26.1x | 8x ~ 46x | WSL2 报告上限 |
| 睿频支持 | 否（接口缺失） | 是 | 由 Windows 主机管理 |
| intel_pstate | 不支持 | 支持（原生 Linux） | WSL2 虚拟化层不暴露 |

**关键发现**：
1. **WSL2 不支持睿频报告**：所有测量结果都被限制在基础频率（2.6 GHz）
2. **但实际支持睿频运行**：CPU 物理上可以睿频到 4.6 GHz
3. **频率控制缺失**：无法通过标准 Linux 接口控制或观察 CPU 频率
4. **性能影响**：虚拟化层有少量性能损失（约 5-10%）

### 5.9 WSL2 环境实测结果（2026年6月）

**测试环境**：
- **平台**：Windows 11 + WSL2 (Ubuntu 24.04)
- **CPU**：13th Gen Intel(R) Core(TM) i5-13420H
- **内核**：6.6.87.2-microsoft
- **CPU架构**：4P + 2E 核心，超线程开启（共12逻辑核心）

**实测数据**（使用 AVX2 向量浮点运算测试）：

| 测试配置 | 绑定核心 | GFLOPS | Windows任务管理器显示频率 |
|---------|---------|--------|---------------------|
| 单核心 P核 | CPU0 | 6.798 | 1.5~1.6 GHz |
| 单核心 P核 | CPU2 | 6.470 | 1.5~1.6 GHz |
| 单核心 P核 | CPU4 | 6.754 | 1.5~1.6 GHz |
| 单核心 P核 | CPU6 | 6.888 | 1.5~1.6 GHz |
| 单核心 E核 | CPU8 | 7.141 | 1.5~1.6 GHz |
| 单核心 E核 | CPU10 | 7.318 | 1.5~1.6 GHz |
| 双 P核 | CPU0,2 | 12.973 | 2.0~2.1 GHz |
| 双 E核 | CPU8,10 | 12.106 | 2.0~2.1 GHz |
| 四 P核 | CPU0,2,4,6 | 20.443 | 2.5~2.6 GHz |
| 全部12核心 | CPU0-11 | 34.174 | 4.5~4.6 GHz |

> **重要说明**：
> - GFLOPS = Giga Floating-point Operations Per Second，本身已包含"/s"含义
> - 之前代码中写的"GFLOPS/s"是重复错误，现已修正为"GFLOPS"
> - 通过 GFLOPS 估算频率的方法不准确，实际频率需通过 Windows 任务管理器观察

**关键发现**：

1. **WSL2 单核心性能受限**：单核心测试时，GFLOPS 仅为 6.5-7.3，远低于理论峰值 36.8 GFLOPS
   - 这是因为 Windows 电源管理策略：单线程负载不足以触发睿频
   - 此时 CPU 运行在低功耗状态（约 1.5-1.6 GHz）

2. **多核心负载触发睿频**：随着核心数量增加，CPU 频率逐步提升
   - 双核心：约 2.0 GHz
   - 四核心：约 2.5-2.6 GHz
   - 全部核心：达到 4.5-4.6 GHz（睿频）

3. **TSC 频率固定**：WSL2 的 TSC 频率是 constant_tsc，始终固定为约 2.6 GHz
   - TSC 无法反映实际运行频率
   - 实际频率只能通过 Windows 任务管理器观察

4. **stress 命令行为一致**：`stress --cpu 4` 使用 4 个线程，触发频率约 2.0-2.1 GHz
   - 这与我们的双核心测试结果相符
   - stress 使用的是整数运算，效率低于浮点运算

**性能分析**：

- **单核心实际频率**：1.5-1.6 GHz（Windows任务管理器显示）
- **多核心实际频率**：最高可达 4.6 GHz（睿频）
- **AVX2 利用率**：单核心约 18-20%，多核心约 93%
- **虚拟化开销**：约 5-10%

**关于频率估算的更正**：

之前使用 `估算频率 = GFLOPS / 8` 的方法是不准确的，原因：
- 这个公式假设 CPU 每个周期能执行 8 次浮点运算（AVX2 理论峰值）
- 但实际执行效率受多种因素影响（指令流水线、缓存命中率、分支预测等）
- WSL2 的 TSC 频率固定，无法通过 RDTSC 测量实际频率
- **正确方法**：在 Windows 任务管理器中观察"当前速度"

**测试工具**：

测试脚本位于 [cpu_freq_test_v3.sh](file:///home/hill/gemm/test/cpu_freq_test_v3.sh)，主要功能：
- 使用 `taskset` 绑定到特定 CPU 核心
- 使用 AVX2 向量指令进行浮点运算
- 通过比较运算次数来判断实际运行频率
- 在运行期间需配合 Windows 任务管理器观察"当前速度"

**结论**：

WSL2 **支持睿频**，但需要足够的负载才能触发。单线程程序难以触发睿频，而多线程程序可以充分利用 CPU 性能。在进行性能测试时，应使用多线程负载来确保 CPU 运行在最高频率。

### 5.10 其他环境测试结果（待补充）

> **说明**：本节用于记录在其他环境上的测试结果，方便对比分析。

#### 测试环境模板

```markdown
**测试环境**：
- **平台**：[Windows/Mac/Linux/WSL]
- **系统**：[Ubuntu 24.04 / macOS Sonoma / ...]
- **CPU**：[型号]
- **内核**：[版本]

**测试结果**：
- 空闲频率：___ GHz
- 满载频率：___ GHz
- 最大倍频：___x
- 睿频支持：[是/否]
- 性能损失：[__%]

**特殊说明**：
[记录该环境下的特殊现象]
```

---

## 六、理论峰值计算

### 6.1 单核心峰值

```
基础频率:  max_gflops = nflops_per_cycle × 1 × GHz_of_processor
                      = 8 × 1 × 2.6 = 20.8 GFLOPS/s

P核睿频:   max_gflops = 8 × 1 × 4.6 = 36.8 GFLOPS/s
```

### 6.2 多核心峰值

```
6核心(基础):  max_gflops = 8 × 6 × 2.6 = 124.8 GFLOPS/s

6核心(睿频):  max_gflops = 8 × 6 × 4.0 ≈ 192 GFLOPS/s
```

---

## 七、参考资料

- [Intel ARK - i5-13420H](https://ark.intel.com/content/www/us/en/ark/products/234892/intel-core-i513420h-processor-12m-cache-up-to-4-60-ghz.html)
- [Intel Turbo Boost Technology](https://www.intel.com/content/www/us/en/architecture-and-technology/turbo-boost/turbo-boost-technology.html)
- [RDTSC Instruction](https://en.wikipedia.org/wiki/Time_Stamp_Counter)
