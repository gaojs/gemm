#!/bin/bash
set -e

IMAGES_DIR="./images"
BASE_DIR="./images"

mkdir -p "$IMAGES_DIR"

echo "========================================"
echo "  Generating adjacent version comparisons"
echo "========================================"

VERSIONS=(
    "MMult4-4x4-avx2"
    "MMult4-4x8-avx2"
    "MMult4-4x8-avx512"
    "MMult4-4x16-avx512"
    "MMult4-4x16-avx512a"
    "MMult4-4x32-avx512"
    "MMult4-4x32-avx512-nc128"
    "MMult4-8x8-avx512"
    "MMult4-8x16-avx512"
    "MMult4-8x24-avx512"
    "MMult4-9x16-avx512"
    "MMult4-10x16-avx512"
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
    
    python3 ./PlotAll.py "$file1" "$file2" "${IMAGES_DIR}/compare_${ver1}_${ver2}.png"
    
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
