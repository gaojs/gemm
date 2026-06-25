#!/bin/bash
set -e

BASELINE="MMult0"
IMAGES_DIR="/home/hill/gemm/images"

mkdir -p "$IMAGES_DIR"

VERSIONS_SSE=(
    "MMult1"
    "MMult2_1x4"
    "MMult2_1x4_3"
    "MMult2_1x4_4"
    "MMult2_1x4_5"
    "MMult2_1x4_6"
    "MMult2_1x4_7"
    "MMult2_1x4_8"
    "MMult3_4x4_3"
    "MMult3_4x4_4"
    "MMult3_4x4_5"
    "MMult3_4x4_6"
    "MMult3_4x4_7"
    "MMult3_4x4_8"
    "MMult3_4x4_9"
    "MMult3_4x4_10"
    "MMult3_4x4_11"
    "MMult3_4x4_12"
    "MMult3_4x4_13"
    "MMult3_4x4_14"
    "MMult3_4x4_15"
)

VERSIONS_AVX2=(
    "MMult4-4x4-avx2"
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

echo ""
echo "========================================"
echo "  All tests completed!"
echo "  Results in: $IMAGES_DIR"
echo "========================================"
ls -la "$IMAGES_DIR/"
