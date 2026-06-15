#!/bin/bash
# Build split-2 (scene0097_00 .. scene0199_00) on scratch: per scene, extract
# color from .sens, make the 100-frame stride-5 subset, then run SAM3 ONCE to
# produce per-instance masks AND derive the per-class masks/ from the instance
# union (class_masks_mode=write). Color + .sens are deleted per scene to keep
# disk/inode use low. Resumable via masks_instance/.complete.
#
#SBATCH --job-name=split2_build
#SBATCH --output=split2_build_%j.log
#SBATCH --error=split2_build_%j.err
#SBATCH --open-mode=append
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4096
#SBATCH --gpus=rtx_4090:1
#SBATCH --tmp=200000
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=niacobone@student.ethz.ch

module purge
module load stack/2024-06 python/3.12.8 cuda/12.8.0 eth_proxy
cd /cluster/scratch/niacobone/sam3
source myenv/bin/activate
set -u

START=${1:-97}
END=${2:-199}
BUILD2=/cluster/scratch/niacobone/scannet_build_split2
SCANS="$BUILD2/scans"
SENSREADER=/cluster/scratch/niacobone/ScanNet/SensReader/python/reader.py
DL=/cluster/scratch/niacobone/sam3/scripts/download_sens.py
PY=/cluster/scratch/niacobone/sam3/myenv/bin/python
PY_SENS=/cluster/scratch/niacobone/ScanNet/myenv/bin/python
INST=/cluster/scratch/niacobone/sam3/scripts/save_instance_masks.py
mkdir -p logs

for i in $(seq "$START" "$END"); do
    S=$(printf "scene%04d_00" "$i")
    SCENE_DIR="$SCANS/$S"
    RAW="$SCENE_DIR/raw_data"
    COLOR="$RAW/color"
    SUBSET="$RAW/subset"
    SENS="$SCENE_DIR/$S.sens"

    if [ -f "$RAW/masks_instance/.complete" ]; then
        echo "[$S] already complete, skipping."; continue
    fi
    echo "===================== $S ====================="

    # 1) ensure .sens (fallback download if the pre-download phase missed it)
    if [ ! -f "$SENS" ]; then
        echo "[$S] .sens missing, downloading..."
        "$PY" "$DL" -o "$BUILD2" --start "$i" --end "$i" --timeout 180 --retries 3 || true
    fi
    if [ ! -f "$SENS" ]; then
        echo "[$S] no .sens available, SKIP scene."; continue
    fi

    # 2) extract color frames
    if [ ! -d "$COLOR" ] || [ -z "$(ls -A "$COLOR" 2>/dev/null)" ]; then
        echo "[$S] extracting color..."
        mkdir -p "$RAW"
        "$PY_SENS" "$SENSREADER" --filename "$SENS" --output_path "$RAW" --export_color_images
    fi

    # 3) rename to zero-padded 5-digit (0.jpg -> 00000.jpg)
    if ls "$COLOR"/[0-9].jpg "$COLOR"/[0-9][0-9].jpg "$COLOR"/[0-9][0-9][0-9].jpg "$COLOR"/[0-9][0-9][0-9][0-9].jpg 2>/dev/null | grep -q .; then
        ( cd "$COLOR" && for f in *.jpg; do n="${f%.jpg}"; nn=$(printf "%05d.jpg" "$((10#$n))"); [ "$f" != "$nn" ] && mv "$f" "$nn"; done )
    fi

    # 4) subset: up to 100 frames at stride 5
    TOTAL=$(ls "$COLOR"/*.jpg 2>/dev/null | wc -l)
    if [ "$TOTAL" -eq 0 ]; then echo "[$S] no color frames, SKIP."; rm -f "$SENS"; continue; fi
    TARGET=$(( (TOTAL + 4) / 5 )); [ "$TARGET" -gt 100 ] && TARGET=100
    if [ ! -d "$SUBSET" ] || [ "$(ls "$SUBSET" 2>/dev/null | wc -l)" -lt "$TARGET" ]; then
        echo "[$S] creating subset ($TARGET of $TOTAL)..."
        mkdir -p "$SUBSET"
        for ((j=0; j<100; j++)); do
            fn=$(printf "%05d.jpg" "$((j*5))")
            [ -f "$COLOR/$fn" ] && cp "$COLOR/$fn" "$SUBSET/"
        done
    fi

    # 5) SAM3: per-instance masks + per-class masks (union) in one pass
    echo "[$S] running SAM3 instance masking..."
    "$PY" "$INST" "$S" \
        --scans_root "$SCANS" \
        --class_masks_mode write \
        --log_file logs/instance_masks_split2.jsonl

    # 6) cleanup heavy inputs (keep subset/ masks/ masks_instance/)
    if [ -f "$RAW/masks_instance/.complete" ]; then
        echo "[$S] cleanup color/ + .sens"
        rm -rf "$COLOR"; rm -f "$SENS"
    else
        echo "[$S] WARNING: no .complete; keeping color/.sens for retry"
    fi
done
echo "split2 build finished (range $START..$END)."
