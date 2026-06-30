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
    "MMult4-4x4-avx2"
    "MMult4-4x8-avx2"
    "MMult4-4x8-avx512"
    "MMult4-8x8-avx512"
    "MMult4-4x16-avx512"
    "MMult4-4x16-avx512a"
    "MMult4-4x32-avx512"
    "MMult4-4x32-avx512-nc128"
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
