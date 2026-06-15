#!/bin/bash
# Fuse split-1 (scene0000..0096) and split-2 (scene0097..0199) into ONE archive.
# Scene names are disjoint, so fusing = unpack both into one tree and re-tar.
# The two source tars are left untouched until the fused tar is verified, so a
# training job reading either split keeps working during the fuse.
#
# Run when no training is actively hammering scratch I/O.
#   bash scripts/fuse_splits.sh
set -euo pipefail

WORK=/cluster/work/igp_psr/niacobone/distillation/dataset/scannet
TAR1="$WORK/scannet_instance_dataset.tar.zst"
TAR2="$WORK/scannet_instance_dataset_split2.tar.zst"
FUSED_STAGE=/cluster/scratch/niacobone/scannet_instance_dataset_full.tar.zst
FUSED_WORK="$WORK/scannet_instance_dataset_full.tar.zst"
TMP=/cluster/scratch/niacobone/fuse_tmp

rm -rf "$TMP" && mkdir -p "$TMP"
echo "[fuse] unpacking split-1 ..."; tar --use-compress-program="zstd -d" -C "$TMP" -xf "$TAR1"
echo "[fuse] unpacking split-2 ..."; tar --use-compress-program="zstd -d" -C "$TMP" -xf "$TAR2"

n_scenes=$(ls "$TMP/scans" | wc -l)
echo "[fuse] combined scenes: $n_scenes"

echo "[fuse] re-taring ..."
tar --use-compress-program="zstd -1 -T0" -C "$TMP" -cf "$FUSED_STAGE" scans
echo "[fuse] fused archive size: $(du -h "$FUSED_STAGE" | cut -f1)"

# verify entry count == sum of the two splits
n_full=$(tar --use-compress-program="zstd -d" -tf "$FUSED_STAGE" | grep -c '\.png$\|\.jpg$' || true)
n1=$(tar --use-compress-program="zstd -d" -tf "$TAR1" | grep -c '\.png$\|\.jpg$' || true)
n2=$(tar --use-compress-program="zstd -d" -tf "$TAR2" | grep -c '\.png$\|\.jpg$' || true)
echo "[fuse] entries: full=$n_full  split1=$n1  split2=$n2  sum=$((n1+n2))"
if [ "$n_full" -ne "$((n1+n2))" ]; then echo "[fuse] ERROR: entry mismatch"; exit 1; fi

cp "$FUSED_STAGE" "$FUSED_WORK.tmp" && mv "$FUSED_WORK.tmp" "$FUSED_WORK"
rm -rf "$TMP"
echo "[fuse] done -> $FUSED_WORK ($(du -h "$FUSED_WORK" | cut -f1))"
echo "[fuse] split tars left in place; delete them once you've adopted the fused tar."
