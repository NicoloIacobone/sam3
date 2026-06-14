#!/bin/bash
# Copy subset/ + masks/ (+ masks_instance/ if present) for all 97 scenes
# from the slow work tree to a clean canonical build tree on fast scratch.
set -e
SRC=/cluster/work/igp_psr/niacobone/distillation/dataset/scannet/scans
DST=/cluster/scratch/niacobone/scannet_build/scans
mkdir -p "$DST"
n=0
for s in $(ls "$SRC"); do
    [ -d "$SRC/$s/raw_data/subset" ] || { echo "SKIP $s (no subset)"; continue; }
    mkdir -p "$DST/$s/raw_data"
    for d in subset masks masks_instance; do
        if [ -d "$SRC/$s/raw_data/$d" ]; then
            rsync -a --delete "$SRC/$s/raw_data/$d/" "$DST/$s/raw_data/$d/"
        fi
    done
    n=$((n+1))
    echo "[$n] staged $s"
done
echo "Staging done: $n scenes -> $DST"
