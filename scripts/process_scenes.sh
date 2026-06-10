#!/bin/bash
# Process ScanNet scenes: download, extract, rename, subset, compute masks

set -e

SCANNET_DIR="/cluster/work/igp_psr/niacobone/distillation/dataset/scannet"
SCANS_DIR="$SCANNET_DIR/scans"
SENSREADER="/cluster/scratch/niacobone/ScanNet/SensReader/python/reader.py"
DOWNLOAD_SCRIPT="/cluster/scratch/niacobone/sam3/scripts/download-scannet.py"
MASK_SCRIPT="/cluster/scratch/niacobone/sam3/scripts/save_text_prompt_masks.py"
PYTHON="/cluster/scratch/niacobone/sam3/myenv/bin/python"
PYTHON_SENSREADER="/cluster/scratch/niacobone/ScanNet/myenv/bin/python"

SCENES=("scene0001_00" "scene0002_00" "scene0003_00" "scene0004_00")

for SCENE in "${SCENES[@]}"; do
    echo "=========================================="
    echo "Processing $SCENE"
    echo "=========================================="

    SCENE_DIR="$SCANS_DIR/$SCENE"
    RAW_DIR="$SCENE_DIR/raw_data"
    COLOR_DIR="$RAW_DIR/color"
    SUBSET_DIR="$RAW_DIR/subset"

    # Step 1: Download
    if [ ! -f "$SCENE_DIR/${SCENE}.sens" ]; then
        echo "[$SCENE] Downloading..."
        printf "\n\n" | $PYTHON "$DOWNLOAD_SCRIPT" -o "$SCANNET_DIR" --id "$SCENE"
    else
        echo "[$SCENE] Already downloaded, skipping."
    fi

    # Step 2: Extract color images
    if [ ! -d "$COLOR_DIR" ] || [ -z "$(ls -A "$COLOR_DIR" 2>/dev/null)" ]; then
        echo "[$SCENE] Extracting color images..."
        mkdir -p "$RAW_DIR"
        $PYTHON_SENSREADER "$SENSREADER" \
            --filename "$SCENE_DIR/${SCENE}.sens" \
            --output_path "$RAW_DIR" \
            --export_color_images
    else
        echo "[$SCENE] Color images already extracted, skipping."
    fi

    # Step 3: Rename images to 5-digit format (0.jpg -> 00000.jpg)
    # Check if any file lacks zero-padding (e.g., "0.jpg" instead of "00000.jpg")
    if ls "$COLOR_DIR"/[0-9].jpg "$COLOR_DIR"/[0-9][0-9].jpg "$COLOR_DIR"/[0-9][0-9][0-9].jpg "$COLOR_DIR"/[0-9][0-9][0-9][0-9].jpg 2>/dev/null | grep -q .; then
        echo "[$SCENE] Renaming images..."
        cd "$COLOR_DIR"
        for file in *.jpg; do
            num="${file%.jpg}"
            new_name=$(printf "%05d.jpg" "$((10#$num))")
            [ "$file" != "$new_name" ] && mv "$file" "$new_name"
        done
    else
        echo "[$SCENE] Images already renamed, skipping."
    fi

    # Step 4: Create subset of 100 images at stride 5
    if [ ! -d "$SUBSET_DIR" ] || [ "$(ls "$SUBSET_DIR" 2>/dev/null | wc -l)" -lt 100 ]; then
        echo "[$SCENE] Creating subset..."
        mkdir -p "$SUBSET_DIR"
        for ((i=0; i<100; i++)); do
            idx=$((i * 5))
            filename=$(printf "%05d.jpg" "$idx")
            cp "$COLOR_DIR/$filename" "$SUBSET_DIR/"
        done
    else
        echo "[$SCENE] Subset already exists, skipping."
    fi

    # Step 5: Compute masks
    if [ ! -d "$RAW_DIR/masks" ] || [ -z "$(ls -A "$RAW_DIR/masks" 2>/dev/null)" ]; then
        echo "[$SCENE] Computing masks..."
        cd /cluster/scratch/niacobone/sam3
        $PYTHON "$MASK_SCRIPT" "$COLOR_DIR"
    else
        echo "[$SCENE] Masks already computed, skipping."
    fi

    echo "[$SCENE] Done!"
done

echo "All scenes processed."
