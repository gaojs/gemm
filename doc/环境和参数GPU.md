# 环境和参数（4090）

## 查询方式

仓库中的示例位于 `cuda-samples/cpp/1_Utilities/deviceQuery/`（示例名称为 `deviceQuery`）。当前环境未安装 CMake，因此使用 CUDA 12.8 的 `nvcc` 直接编译：

```bash
cd /home/hill/gemm/cuda-samples
nvcc -std=c++17 -I Common cpp/1_Utilities/deviceQuery/deviceQuery.cpp -o /tmp/deviceQuery
/tmp/deviceQuery
```

查询结果为 `Result = PASS`。

## 软件环境

| 参数 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 D |
| CUDA Driver Version | 13.2 |
| CUDA Runtime Version | 12.8 |
| CUDA Capability | 8.9（SM 89） |
| CUDA 设备数量 | 1 |
| PCI Domain / Bus / Device | 0 / 129 / 0 |

## GPU 与内存参数

| 参数 | 值 |
|---|---:|
| 全局显存 | 25,252,724,736 bytes（24,083 MiB，约 23.5 GiB） |
| SM 数量 | 114 |
| CUDA Cores / SM | 128 |
| CUDA Core 总数 | 14,592 |
| GPU 最大时钟 | 2,520 MHz（2.52 GHz） |
| 显存时钟 | 10,501 MHz |
| 显存总线宽度 | 384-bit |
| L2 Cache | 75,497,472 bytes（72 MiB） |
| 常量内存 | 65,536 bytes（64 KiB） |
| 每个 SM 共享内存 | 102,400 bytes（100 KiB） |
| 每个 block 共享内存 | 49,152 bytes（48 KiB） |
| 每个 block 可用寄存器数 | 65,536 |
| ECC | Disabled |

## 线程与网格限制

| 参数 | 值 |
|---|---|
| Warp 大小 | 32 threads |
| 每个 SM 最大线程数 | 1,536 |
| 每个 block 最大线程数 | 1,024 |
| Block 最大维度（x, y, z） | (1024, 1024, 64) |
| Grid 最大维度（x, y, z） | (2147483647, 65535, 65535) |

## 其他设备能力

| 能力 | 结果 |
|---|---|
| 并发内存拷贝与 kernel 执行 | 支持，2 个 copy engines |
| Kernel 执行时间限制 | 无 |
| 集成 GPU | 否 |
| Host page-locked memory 映射 | 支持 |
| Unified Addressing（UVA） | 支持 |
| Managed Memory | 支持 |
| Compute Preemption | 支持 |
| Cooperative Kernel Launch | 支持 |
| Compute Mode | Default |

## 纹理尺寸限制

| 纹理类型 | 最大尺寸 |
|---|---|
| 1D | 131,072 |
| 2D | 131,072 × 65,536 |
| 3D | 16,384 × 16,384 × 16,384 |
| Layered 1D | 32,768，最多 2,048 层 |
| Layered 2D | 32,768 × 32,768，最多 2,048 层 |
| 最大 memory pitch | 2,147,483,647 bytes |
| Texture alignment | 512 bytes |

> 这些数据来自当前运行环境中的 `deviceQuery` 实测输出；时钟频率等参数可能随设备状态变化。
