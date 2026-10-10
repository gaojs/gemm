#!/bin/bash
# 首次配置本项目 Python 绘图环境时，在普通 Terminal 中手动执行：
#   cd /home/hill/gemm
#   sudo apt-get install -y python3.12-venv
#   python3 -m venv .venv
#   .venv/bin/python -m pip install matplotlib
#   .venv/bin/python -c "import matplotlib; print(matplotlib.__version__)"
# 上述步骤只需执行一次；运行基准时不会自动安装系统包或 Python 包。
set -euo pipefail
cd "$(dirname "$0")"

NVCC=$(command -v nvcc)
GPU_INFO=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader)
GPU_INFO=${GPU_INFO%%$'\n'*}
case "$GPU_INFO" in
    *A100*,*8.0*) ARCH=sm_80; OUTPUT=fig-gpu-a100 ;;
    *"RTX 4090 D"*,*8.9*) ARCH=sm_89; OUTPUT=fig-gpu-4090 ;;
    *) echo "不支持的 GPU 或计算能力：$GPU_INFO" >&2; exit 1 ;;
esac

PYTHON=./.venv/bin/python
if ! test -x "$PYTHON" || ! "$PYTHON" -c 'import matplotlib' 2>/dev/null; then
    echo "请先在本项目的 .venv 中安装 matplotlib，再重新运行。" >&2
    exit 1
fi

mkdir -p "$OUTPUT"
BINARY=$(mktemp /tmp/gpu_gemm.XXXXXX)
RESULTS_TMP=$(mktemp "$OUTPUT/gpu_results.m.XXXXXX")
trap 'rm -f "$BINARY" "$RESULTS_TMP"' EXIT
"$NVCC" -O3 -arch="$ARCH" gpu_gemm.cu -lcublas -lcudart -o "$BINARY"
"$BINARY" > "$RESULTS_TMP"
"$PYTHON" gpu_plot_results.py "$RESULTS_TMP" "$OUTPUT/gpu_comparison.png"
mv "$RESULTS_TMP" "$OUTPUT/gpu_results.m"
echo "结果已保存至 $OUTPUT/"
