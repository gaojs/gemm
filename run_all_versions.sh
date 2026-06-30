#!/bin/bash
set -e

BASELINE="MMult0"
IMAGES_DIR="/home/hill/gemm/images"

mkdir -p "$IMAGES_DIR"

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

VERSIONS_AVX512=(
    "MMult4-4x8-avx512"
    "MMult4-8x8-avx512"
    "MMult4-4x16-avx512"
    "MMult4-4x16-avx512a"
    "MMult4-4x32-avx512"
    "MMult4-4x32-avx512-nc128"
)

echo "========================================"
echo "  Running baseline: $BASELINE"
echo "========================================"
make OLD=$BASELINE NEW=$BASELINE run
cp "output_${BASELINE}.m" "$IMAGES_DIR/"

echo ""
echo "========================================"
echo "  Generating baseline plot"
echo "========================================"
python3 PlotAll.py
cp "compare_${BASELINE}_${BASELINE}.png" "$IMAGES_DIR/"

for ver in "${VERSIONS_SSE[@]}"; do
    echo ""
    echo "========================================"
    echo "  Testing: $ver"
    echo "========================================"
    make OLD=$BASELINE NEW="$ver" run
    cp "output_${ver}.m" "$IMAGES_DIR/"
    python3 PlotAll.py
    png_file="compare_${BASELINE}_${ver}.png"
    if [ -f "$png_file" ]; then
        cp "$png_file" "$IMAGES_DIR/"
        echo "Saved: $IMAGES_DIR/$png_file"
    fi
done

for ver in "${VERSIONS_AVX2[@]}"; do
    echo ""
    echo "========================================"
    echo "  Testing: $ver (AVX2)"
    echo "========================================"
    make OLD=$BASELINE NEW="$ver" CFLAGS="-O2 -Wall -mavx2 -mfma" run
    cp "output_${ver}.m" "$IMAGES_DIR/"
    python3 PlotAll.py
    png_file="compare_${BASELINE}_${ver}.png"
    if [ -f "$png_file" ]; then
        cp "$png_file" "$IMAGES_DIR/"
        echo "Saved: $IMAGES_DIR/$png_file"
    fi
done

for ver in "${VERSIONS_AVX512[@]}"; do
    echo ""
    echo "========================================"
    echo "  Testing: $ver (AVX-512)"
    echo "========================================"
    make OLD=$BASELINE NEW="$ver" CFLAGS="-O2 -Wall -mavx512f -mavx512dq -mfma" run
    cp "output_${ver}.m" "$IMAGES_DIR/"
    python3 PlotAll.py
    png_file="compare_${BASELINE}_${ver}.png"
    if [ -f "$png_file" ]; then
        cp "$png_file" "$IMAGES_DIR/"
        echo "Saved: $IMAGES_DIR/$png_file"
    fi
done

echo ""
echo "========================================"
echo "  All tests completed!"
echo "  Results in: $IMAGES_DIR"
echo "========================================"
ls -la "$IMAGES_DIR/"
