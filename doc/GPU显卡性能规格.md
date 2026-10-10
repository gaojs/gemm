# A100 SXM4 80GB 与 RTX 4090 D：性能规格及 DGEMM 实测对比

## 先说结论

**A100 并非在所有任务上都比 RTX 4090 D 快；但本项目的** **`double`** **双精度矩阵乘法（FP64 DGEMM）是 A100 的强项。** RTX 4090 D 的单精度/低精度图形与 AI 算力很强，不能拿它的 FP32 Shader TFLOPS 去与 A100 的 FP64 DGEMM 比较。本项目固定 n=1024 时，A100 的 cuBLAS 约 **13,498 GFLOPS**，RTX 4090 D 约 **902 GFLOPS**，实测约 **15.0 倍**；这是两个环境中这项工作负载的结果，不是“所有程序 A100 都快 15 倍”。

## 规格（区分官方标称与推算）

| 项目                      | NVIDIA A100-SXM4-80GB |                   GeForce RTX 4090 D | 口径                                                |
| ----------------------- | --------------------: | -----------------------------------: | ------------------------------------------------- |
| 产品定位 / 架构               |    数据中心 HPC/AI；Ampere |        GeForce 游戏/创作/AI；Ada Lovelace | 官方产品信息                                            |
| CUDA Compute Capability |                   8.0 |                                  8.9 | 本项目测试程序设备报告                                       |
| 显存                      |           80 GB HBM2e |                         24 GB GDDR6X | 官方标称；程序实际可见约 79.3 / 23.5 GiB                      |
| 显存带宽                    |            2,039 GB/s |                            此处不提供官方数值 | A100 SXM 官方数据表；不要用显存容量推断带宽                        |
| FP32/Shader 计算          |      FP32：19.5 TFLOPS |                   Shader：约 74 TFLOPS | 两项均为官方标称、近似单精度口径，**不是 FP64**                      |
| 普通 FP64                 |            9.7 TFLOPS | 约 1.15 TFLOPS（**估算值，非 NVIDIA 官网标称**） | A100 官方数据表；4090 D 按 FP32 约 73.54 TFLOPS × 1/64 估算 |
| FP64 Tensor Core        |           19.5 TFLOPS |      不应套用 A100 的 FP64 Tensor Core 峰值 | A100 官方数据表；是否实际用到取决于软件 kernel                     |
| 最大板卡功耗                  |                 400 W |                                425 W | 官方额定值，不是测试时实际功耗                                   |

<br />

来源：[NVIDIA A100 80GB 数据表](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf)、[NVIDIA GeForce RTX 4090 D 官方规格](https://www.nvidia.cn/geforce/graphics-cards/compare/)、[4090 D FP64 推算参考](https://fpsbench.com/gpu/nvidia-geforce-rtx-4090-d)。NVIDIA 的 GeForce 规格表没有直接列出 4090 D 的 FP64 峰值，因此上面的 1.15 TFLOPS 是根据常见 1:64 比例和标称 FP32 估算、并得到第三方规格交叉印证的**近似理论值**；并非一个单独从官网抄来的认证指标。官方 Shader 74 TFLOPS 的定义亦不等于实际 `cublasSgemm` 基准成绩。

## 本项目实测（GFLOPS）

测试代码在 `gpu_gemm.cu` 中使用 `double` 输入和输出、`cublasDgemm`；每个尺寸先预热一次，之后 CUDA event 测量 5 次调用的平均 GPU 耗时，以 `2n³ / 耗时` 计算 GFLOPS。表格里**同一行、同一 n** 才是直接可比的数字。

| n / 实现                   |              A100 |    RTX 4090 D | A100 / 4090 D |
| ------------------------ | ----------------: | ------------: | ------------: |
| 1024 / cuBLAS DGEMM      |         13,498.44 |        901.94 |        14.97× |
| 1024 / 手写 `cuda-reg-2x2` |          3,917.99 |      1,130.18 |         3.47× |
| 1024 / 手写 `cuda-naive`   |          1,829.94 |      1,104.81 |         1.66× |
| 两端 cuBLAS 各自最好的**不同尺寸**  | 14,088.57（n=1152） | 988.67（n=960） | 不适合直接作为同尺寸加速比 |

数据文件：[A100 原始结果](../fig-gpu-a100/gpu_results.m) 与 [4090 D 原始结果](../fig-gpu-4090/gpu_results.m)。如果看到“13,948 GFLOPS”，请注意本项目文件中的**n=1024 是 13,498.44 GFLOPS**；所有已测点中最高为 **14,088.57 GFLOPS**，没有 13,948 GFLOPS 这一记录。

## 为什么 cuBLAS 的差距比手写 kernel 大？

1. **精度和硬件设计不同。** A100 提供 9.7 TFLOPS 普通 FP64 和 19.5 TFLOPS FP64 Tensor Core 标称能力；4090 D 对应普通 FP64 推算仅约 1.15 TFLOPS。其 74 TFLOPS 是 Shader/近似单精度指标，不能用于 DGEMM。A100 的 13.50 TFLOPS 已高于其普通 FP64 标称 9.7 TFLOPS，与利用加速路径相符，但**仅凭结果不能证实具体使用了 FP64 Tensor Core**，需要查看 cuBLAS 实际 kernel 和 Nsight 指令指标。
2. **同一库在两台卡上会选择不同的实现。** `cublasDgemm` 内部 kernel 和配置随架构、尺寸、库版本变化；本项目并未固定内部算法。4090 D 的 902 GFLOPS 低于估算的约 1.15 TFLOPS FP64 峰值，但不意味着 cuBLAS 有错误：算法/访存/尺寸/时钟/测试噪声都可能影响数值，原因需要 profiler 实证。
3. **本项目矩阵规模不大。** n 从 64 到 1280；GPU kernel/库选择与发射开销相对于计算的比例，以及缓存驻留状况都可能影响名次。五次计时且未做多轮统计；这些数据不应被解读成两张卡在所有尺寸或所有软件版本下的绝对极限。

对于 FP32/TF32/FP16、游戏、推理、需要 >24 GB 显存的任务，结论需分别重新测量，不能由这份 FP64 DGEMM 结果外推。要判定为何 A100 cuBLAS 具体快、4090 D cuBLAS 输给手写 kernel，可按 [Nsight 使用说明](Nsight使用说明.md) 在相同尺寸下查 GPU 时间线、实际 kernel、FP64/Tensor 指令、时钟及访存指标。
