#!/bin/bash
# AMD EPYC 7H12 (Zen 2, AVX2 only) GEMM benchmark script
# Reproduces the GEMM optimization project on AMD EPYC environment
# Results saved to ./ae7/ (ae7 = AMD EPYC 7H12)
#
# Features:
#   - Resume support: skips versions whose output .m already exists in ae7/
#   - Use -f flag to force re-run all versions:  ./run_all_avx2.sh -f

BASELINE="MMult0"
RESULTS_DIR="./ae7"

# Python interpreter with matplotlib support (conda python + local pylibs symlinks)
export PYTHONPATH="/home/hill/gemm/pylibs"
PYTHON="/opt/conda/bin/python"

# AMD EPYC 7H12 theoretical peak parameters (AVX2 only)
#   2 FMA units x 4 double x 2 ops = 16 FLOPs/cycle
#   Max boost: 3.3 GHz
#   Single-core peak: 16 x 3.3 = 52.8 GFLOPS
export NFLOPS_PER_CYCLE=16
export NPROCESSORS=1
export GHZ_OF_PROCESSOR=3.3

FORCE=0
if [ "$1" = "-f" ]; then FORCE=1; fi

mkdir -p "$RESULTS_DIR"

# Versions to test (SSE + AVX + AVX2; NO AVX-512 since AMD EPYC 7H12 doesn't support it)
VERSIONS_SSE=(
    "MMult1-fn"
    "MMult2-1x4-2"
    "MMult2-1x4-3-fn"
    "MMult2-1x4-4-inline"
    "MMult2-1x4-5-merge"
    "MMult2-1x4-6-reg"
    "MMult2-1x4-7-ptr"
    "MMult2-1x4-8-unroll"
    "MMult3-4x4-3-fn"
    "MMult3-4x4-4-inline"
    "MMult3-4x4-5-merge"
    "MMult3-4x4-6-reg"
    "MMult3-4x4-7-ptr"
    "MMult3-4x4-8-reg"
    "MMult3-4x4-9-rearr"
    "MMult3-4x4-10-avx"
    "MMult3-4x4-11-block"
    "MMult3-4x4-12-packa"
    "MMult3-4x4-13-packa"
    "MMult3-4x4-14-packb"
    "MMult3-4x4-15-packb"
)

VERSIONS_AVX2=(
    "MMult4-4x4-avx2"
    "MMult4-4x8-avx2"
)

# Helper: run one version (skip if already done and not forced)
run_version() {
    local ver="$1"
    local flags="${2:-}"
    local out_file="$RESULTS_DIR/output_${ver}.m"

    if [ $FORCE -eq 0 ] && [ -f "$out_file" ]; then
        echo "  [SKIP] $ver (already done: $out_file)"
        return 0
    fi

    echo ""
    echo "========================================"
    echo "  Testing: $ver"
    echo "========================================"
    if [ -n "$flags" ]; then
        make OLD=$BASELINE NEW="$ver" CFLAGS="$flags" run
    else
        make OLD=$BASELINE NEW="$ver" run
    fi
    cp "output_${ver}.m" "$RESULTS_DIR/"
    if [ -f "compare_${BASELINE}_${ver}.png" ]; then
        cp "compare_${BASELINE}_${ver}.png" "$RESULTS_DIR/"
    fi
    echo "  [DONE] $ver"
}

echo "========================================"
echo "  Environment: AMD EPYC 7H12 (AVX2 only)"
echo "  Theoretical peak: 52.8 GFLOPS (single core)"
echo "  Results directory: $RESULTS_DIR"
echo "  Resume mode: $([ $FORCE -eq 0 ] && echo 'ON (skip done)' || echo 'OFF (force all)')"
echo "========================================"

# --- Step 1: Run baseline (MMult0) ---
run_version "$BASELINE"

# --- Step 2: Run SSE/AVX versions ---
for ver in "${VERSIONS_SSE[@]}"; do
    run_version "$ver"
done

# --- Step 3: Run AVX2 versions (explicit -mavx2 -mfma flags) ---
for ver in "${VERSIONS_AVX2[@]}"; do
    run_version "$ver" "-O2 -Wall -mavx2 -mfma -g"
done

# --- Step 4: Generate AVX2 internal comparison plot ---
echo ""
echo "========================================"
echo "  Generating AVX2 versions comparison plot"
echo "========================================"
if [ -f "$RESULTS_DIR/output_MMult4-4x4-avx2.m" ] && [ -f "$RESULTS_DIR/output_MMult4-4x8-avx2.m" ]; then
    $PYTHON PlotAll.py \
        "$RESULTS_DIR/output_MMult4-4x4-avx2.m" \
        "$RESULTS_DIR/output_MMult4-4x8-avx2.m" \
        "$RESULTS_DIR/compare_avx2_versions.png"
else
    echo "  [SKIP] AVX2 .m files not found, skip plot"
fi

# --- Step 5: Generate summary plot (baseline vs AVX2 best) ---
echo ""
echo "========================================"
echo "  Generating summary: MMult0 vs MMult4-4x8-avx2"
echo "========================================"
if [ -f "$RESULTS_DIR/output_${BASELINE}.m" ] && [ -f "$RESULTS_DIR/output_MMult4-4x8-avx2.m" ]; then
    $PYTHON PlotAll.py \
        "$RESULTS_DIR/output_${BASELINE}.m" \
        "$RESULTS_DIR/output_MMult4-4x8-avx2.m" \
        "$RESULTS_DIR/compare_baseline_vs_avx2_best.png"
else
    echo "  [SKIP] Required .m files not found, skip plot"
fi

# --- Step 6: Generate optimization progression summary plot ---
echo ""
echo "========================================"
echo "  Generating progression: MMult1-fn vs MMult4-4x8-avx2"
echo "========================================"
if [ -f "$RESULTS_DIR/output_MMult1-fn.m" ] && [ -f "$RESULTS_DIR/output_MMult4-4x8-avx2.m" ]; then
    $PYTHON PlotAll.py \
        "$RESULTS_DIR/output_MMult1-fn.m" \
        "$RESULTS_DIR/output_MMult4-4x8-avx2.m" \
        "$RESULTS_DIR/compare_progression_mmult1_to_avx2_best.png"
else
    echo "  [SKIP] Required .m files not found, skip plot"
fi

echo ""
echo "========================================"
echo "  All tests completed!"
echo "  Results saved in: $RESULTS_DIR"
echo "========================================"
echo ""
echo "Performance data files (.m):"
ls -1 "$RESULTS_DIR/"*.m 2>/dev/null | wc -l
echo "Comparison plots (.png):"
ls -1 "$RESULTS_DIR/"*.png 2>/dev/null | wc -l
echo ""
echo "=== Files in $RESULTS_DIR ==="
ls -la "$RESULTS_DIR/"
