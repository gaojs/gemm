#!/bin/bash
set -e

IMAGES_DIR="/home/hill/gemm/images"
BASE_DIR="/home/hill/gemm/images"

mkdir -p "$IMAGES_DIR"

echo "========================================"
echo "  Generating adjacent version comparisons"
echo "========================================"

VERSIONS=(
    "MMult0"
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
    "MMult4-4x4-avx2"
)

for ((i=0; i<${#VERSIONS[@]}-1; i++)); do
    ver1="${VERSIONS[i]}"
    ver2="${VERSIONS[i+1]}"
    
    file1="${BASE_DIR}/output_${ver1}.m"
    file2="${BASE_DIR}/output_${ver2}.m"
    
    if [ ! -f "$file1" ]; then
        echo "Warning: $file1 not found, skipping..."
        continue
    fi
    
    if [ ! -f "$file2" ]; then
        echo "Warning: $file2 not found, skipping..."
        continue
    fi
    
    echo ""
    echo "Comparing: $ver1 vs $ver2"
    
    python3 /home/hill/gemm/PlotAll.py "$file1" "$file2" "${IMAGES_DIR}/compare_${ver1}_${ver2}.png"
    
    echo "Saved: compare_${ver1}_${ver2}.png"
done

echo ""
echo "========================================"
echo "  All adjacent comparisons completed!"
echo "  Results in: $IMAGES_DIR"
echo "========================================"
echo ""
echo "Total comparisons: $((${#VERSIONS[@]}-1))"
ls -la "${IMAGES_DIR}/compare_*.png" | wc -l
