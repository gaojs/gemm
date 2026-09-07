#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p fig-gpu
NVCC=/usr/local/cuda/bin/nvcc
CUDA_INC=/usr/local/cuda/targets/x86_64-linux/include
CUDA_LIB=/usr/local/cuda/targets/x86_64-linux/lib
# Python 解释器：项目 .venv（已安装 matplotlib/numpy）
PYTHON=/home/hill/gemm/.venv/bin/python
$NVCC -O3 -arch=sm_80 -I"$CUDA_INC" gpu_gemm.cu -L"$CUDA_LIB" -lcublas -lcudart -o gpu_gemm.x
RESULTS_TMP="fig-gpu/gpu_results.m.tmp"
trap 'rm -f "$RESULTS_TMP"' EXIT
if ! LD_LIBRARY_PATH="$CUDA_LIB:${LD_LIBRARY_PATH:-}" ./gpu_gemm.x > "$RESULTS_TMP"; then
    echo "GPU GEMM 运行失败，未覆盖已有结果文件。" >&2
    exit 1
fi
mv "$RESULTS_TMP" fig-gpu/gpu_results.m
$PYTHON gpu_plot_results.py fig-gpu/gpu_results.m fig-gpu/gpu_comparison.png
