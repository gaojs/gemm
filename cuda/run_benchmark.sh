#!/bin/bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
NVCC=${NVCC:-$(command -v nvcc)}
if [[ ! "${ARCH:-}" =~ ^sm_[0-9]+$ ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ' .')
    CAP=${CAP#compute_cap:}
    ARCH="sm_${CAP:-75}"
  else
    ARCH=sm_75
  fi
fi
BINARY=$(mktemp /tmp/gemm_runner.XXXXXX)
RESULTS_TMP=$(mktemp /tmp/gemm_results.XXXXXX)
trap 'rm -f "$BINARY" "$RESULTS_TMP"' EXIT
"$NVCC" -O3 -std=c++14 -arch="$ARCH" cuda/gpu_runner.cu -lcublas -lcudart -o "$BINARY"
if [[ $# -gt 0 ]]; then
  "$BINARY" "$1" | tee "$RESULTS_TMP"
else
  "$BINARY" | tee "$RESULTS_TMP"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_INFO=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -n1)
  case "$GPU_INFO" in
    *A100*,*8.0*) OUTPUT="$ROOT/fig-gpu-a100" ;;
    *"RTX 4090 D"*,*8.9*) OUTPUT="$ROOT/fig-gpu-4090" ;;
    *) OUTPUT="$ROOT/fig-gpu" ;;
  esac
  if [[ -x "$ROOT/.venv/bin/python" ]] && "$ROOT/.venv/bin/python" -c 'import matplotlib' 2>/dev/null; then
    mkdir -p "$OUTPUT"
    "$ROOT/.venv/bin/python" "$ROOT/cuda/plot_results.py" "$RESULTS_TMP" "$OUTPUT/gpu_comparison.png"
    cp "$RESULTS_TMP" "$OUTPUT/gpu_results.m"
    echo "结果已保存至 $OUTPUT/"
  fi
fi
