#!/bin/bash
# Process ScanNet scenes: download, extract, rename, subset, compute masks
#
# Specify job name.
#SBATCH --job-name=process_scenes
#
# Specify output file.
#SBATCH --output=process_scenes_%j.log
#
# Specify error file.
#SBATCH --error=process_scenes_%j.err
#
# Specify open mode for log files.
#SBATCH --open-mode=append
#
# Specify time limit.
#SBATCH --time=08:00:00
#
# Specify number of tasks.
#SBATCH --ntasks=1
#
# Specify number of CPU cores per task.
#SBATCH --cpus-per-task=8
#
# Specify memory limit per CPU core.
#SBATCH --mem-per-cpu=4096
#
# Specify number of required GPUs.
#SBATCH --gpus=rtx_4090:1
#
# Specify disk limit on local scratch.
#SBATCH --tmp=500000
#
# Specify email notifications.
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=niacobone@student.ethz.ch
#

module purge
module load stack/2024-06 python/3.12.8 cuda/12.8.0 eth_proxy
cd /cluster/scratch/niacobone/sam3
source myenv/bin/activate

set -e

SCANNET_DIR="/cluster/work/igp_psr/niacobone/distillation/dataset/scannet"
SCANS_DIR="$SCANNET_DIR/scans"
SENSREADER="/cluster/scratch/niacobone/ScanNet/SensReader/python/reader.py"
DOWNLOAD_SCRIPT="/cluster/scratch/niacobone/sam3/scripts/download-scannet.py"
MASK_SCRIPT="/cluster/scratch/niacobone/sam3/scripts/save_text_prompt_masks.py"
PYTHON="/cluster/scratch/niacobone/sam3/myenv/bin/python"
PYTHON_SENSREADER="/cluster/scratch/niacobone/ScanNet/myenv/bin/python"

SCENES=()
for i in $(seq 81 150); do
    SCENES+=("$(printf "scene%04d_00" $i)")
done

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
        yes '' | $PYTHON "$DOWNLOAD_SCRIPT" -o "$SCANNET_DIR" --id "$SCENE"
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

    # Step 4: Create subset of up to 100 images at stride 5
    # If the scene has fewer than 500 frames, take as many as available (at most 100).
    TOTAL_IMAGES=$(ls "$COLOR_DIR"/*.jpg 2>/dev/null | wc -l)
    TARGET_COUNT=$(( (TOTAL_IMAGES + 4) / 5 ))   # frames available at stride 5
    [ "$TARGET_COUNT" -gt 100 ] && TARGET_COUNT=100
    if [ ! -d "$SUBSET_DIR" ] || [ "$(ls "$SUBSET_DIR" 2>/dev/null | wc -l)" -lt "$TARGET_COUNT" ]; then
        echo "[$SCENE] Creating subset ($TARGET_COUNT images from $TOTAL_IMAGES available)..."
        mkdir -p "$SUBSET_DIR"
        for ((i=0; i<100; i++)); do
            idx=$((i * 5))
            filename=$(printf "%05d.jpg" "$idx")
            # Only copy frames that actually exist; stop once we run out.
            if [ -f "$COLOR_DIR/$filename" ]; then
                cp "$COLOR_DIR/$filename" "$SUBSET_DIR/"
            fi
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

    # Step 6: Clean up unused files (keep only raw_data)
    echo "[$SCENE] Cleaning up unused files..."
    find "$SCENE_DIR" -maxdepth 1 -mindepth 1 ! -name "raw_data" -exec rm -rf {} +

    echo "[$SCENE] Done!"
done

echo "All scenes processed."
