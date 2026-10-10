# GPU GEMM 优化复现记录

## 1. 目标与实现

本项目使用 `gpu_gemm.cu` 测试双精度矩阵乘法 `C = A × B`，与 CPU 版本分开构建。包含 `cuda-naive`、32×32 共享内存分块 `cuda-tiled-32`、寄存器分块 `cuda-reg-2x2` / `cuda-reg-4x4` / `cuda-reg-4x4-64`，以及 `cublas-dgemm`。测试矩阵规模为 64、128、……、1280；每个版本输出 GFLOPS 和相对 CPU 参考结果的最大绝对误差。

## 2. 当前工具链与运行方式

当前 RTX 4090 D（计算能力 8.9）的普通 Terminal 使用 `/usr/bin/nvcc`（CUDA Toolkit 12.0），不是先前 A100 环境的 `/usr/local/cuda/bin/nvcc`（CUDA Toolkit 13.2）。`nvidia-smi` 显示的 CUDA 13.2 是驱动支持的版本，不能据此推断当前安装了 Toolkit 13.2。当前环境的 `cicc` 位于 `/usr/lib/nvidia-cuda-toolkit/bin/cicc`，由 `nvcc` 自动调用，不需要加入 `PATH`；不能再用旧文档的 `/usr/local/cuda/nvvm/bin/cicc` 路径判断工具链是否完整。

在**可访问 GPU 的普通 Terminal** 中先确认环境；完成下方项目 `.venv` 准备后，再运行脚本：

```bash
cd /home/hill/gemm
nvcc -V
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```

脚本依据设备名和计算能力选择 `sm_89`（RTX 4090 D）或 `sm_80`（A100），调用当前 `PATH` 中的 `nvcc` 编译，运行后仅使用项目目录下的 `.venv/bin/python` 和 matplotlib 绘图，不依赖其他项目的 Python 环境。当前环境已经在项目 `.venv` 中安装了 matplotlib，无需再安装 `uv`。当前 4090 D 环境中，默认库搜索路径可能加载到旧版 `libcuda.so.1`（535），而 `nvidia-smi` 显示的驱动是 595；直接运行脚本会在 `cudaGetDeviceProperties` 处报 `system has unsupported display driver / cuda driver combination`。普通 Terminal 中已验证：临时将与驱动匹配的 `/usr/lib64` 放在库搜索路径前面，可以成功生成结果和图片。请这样运行（`>` 是 shell 的续行提示符，不要输入）：

```bash
cd /home/hill/gemm
.venv/bin/python -c "import matplotlib; print(matplotlib.__version__)"
LD_LIBRARY_PATH=/usr/lib64:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-} \
  ./gpu_run_benchmark.sh
```

该环境变量仅对这一条命令生效，无需升级 CUDA Toolkit；不要把旧版驱动库路径排在 `/usr/lib64` 之前。

如果将来在新环境中重建依赖，可按脚本开头记录的首次安装命令逐条执行：

```bash
cd /home/hill/gemm
sudo apt-get install -y python3.12-venv
python3 -m venv .venv
.venv/bin/python -m pip install matplotlib
.venv/bin/python -c "import matplotlib; print(matplotlib.__version__)"
```

这些命令不会由基准脚本自动执行；还需确保 `nvcc`、`nvidia-smi` 和 CUDA/cuBLAS 开发库可用。脚本不支持其他 GPU，并只使用列表中的第一张 GPU；使用多 GPU 环境时，请确认设备选择一致。

| GPU | 编译目标 | 结果目录 |
|---|---|---|
| NVIDIA A100 | `sm_80` | `fig-gpu-a100/` |
| NVIDIA GeForce RTX 4090 D | `sm_89` | `fig-gpu-4090/` |

每个目录的 `gpu_results.m` 保存六种实现的实测结果，`gpu_comparison.png` 为对比图。脚本运行失败时不会覆盖已有的结果文件；成功执行则会更新当前 GPU 对应的结果与图片。**不要在当前 4090 D 环境运行时把 A100 的历史结果当作新生成的数据。** 若 Trae 命令执行沙箱拦截 `/proc` 或 `/dev/nvidia*`，请在没有该限制的普通 Terminal 中运行；沙箱内能够编译不代表能够执行 GPU benchmark。

## 3. RTX 4090 D 结果的原始生成方式

`fig-gpu-4090/gpu_results.m` 最初是在当前环境的普通 Terminal 中使用以下命令生成的；它不是旧版固定 A100 参数的 `gpu_run_benchmark.sh` 所生成：

```bash
cd /home/hill/gemm
mkdir -p fig-gpu-4090
/usr/bin/nvcc -O3 -arch=sm_89 -I/usr/include gpu_gemm.cu \
  -L/usr/lib/x86_64-linux-gnu -lcublas -lcudart \
  -o /tmp/gpu_gemm_4090.x
LD_LIBRARY_PATH=/usr/lib64:/usr/local/cuda/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu \
  /tmp/gpu_gemm_4090.x > fig-gpu-4090/gpu_results.m
```

之后使用 `gpu_plot_results.py` 读取结果生成图片。更新后的脚本使用当前工具链编译，目标仍是 `sm_89`、同一个 `gpu_gemm.cu`；在此 4090 D 环境中运行时需按上文指定库搜索路径。由于重新运行会重新测量，不保证 GFLOPS 与原结果逐位相同。

## 4. 历史数据和判读

A100 的旧版测试在 CUDA Toolkit 13.2、计算能力 8.0 的环境中进行；原始结果保存在 `fig-gpu-a100/`，不要混同于当前的 4090 D。双精度 cuBLAS 在 A100 上明显领先手写 kernel；在 4090 D 上并非最快。固定 n=1024 时，A100 cuBLAS 为约 13,498 GFLOPS，4090 D cuBLAS 为约 902 GFLOPS。应先核对误差是否在双精度浮点计算的正常范围，再比较相同规模下的性能；不能预设 tiled 必然比 naive 快或 cuBLAS 在每个 GPU 上都是最快。

硬件规格与差异分析见 [GPU 显卡性能规格](GPU显卡性能规格.md)，性能分析步骤见 [Nsight 使用说明](Nsight使用说明.md)。
