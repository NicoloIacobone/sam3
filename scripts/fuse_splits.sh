#!/bin/bash
# Fuse split-1 (scene0000..0096) and split-2 (scene0097..0199) into ONE archive.
#
# We tar DIRECTLY from the two on-scratch build trees (disjoint scene names),
# NOT by unpacking the two tars. Unpacking would create ~830K temp files and
# blow the scratch inode quota; tarring from the existing trees creates only the
# single output archive. GNU tar supports multiple -C segments in one create,
# and both "scans" entries merge on extraction.
#
# The two split tars in work are left untouched until the fused tar is verified.
set -euo pipefail

B1=/cluster/scratch/niacobone/scannet_build           # scans/scene0000..0096
B2=/cluster/scratch/niacobone/scannet_build_split2    # scans/scene0097..0199
WORK=/cluster/work/igp_psr/niacobone/distillation/dataset/scannet
FUSED_STAGE=/cluster/scratch/niacobone/scannet_instance_dataset_full.tar.zst
FUSED_WORK="$WORK/scannet_instance_dataset_full.tar.zst"

s1=$(ls "$B1/scans" | wc -l); s2=$(ls "$B2/scans" | wc -l)
echo "[fuse] split-1 scenes=$s1  split-2 scenes=$s2  expected total=$((s1+s2))"

echo "[fuse] taring both trees into one archive (zstd -1) ..."
tar --use-compress-program="zstd -1 -T0" -cf "$FUSED_STAGE" \
    -C "$B1" scans \
    -C "$B2" scans
echo "[fuse] fused archive size: $(du -h "$FUSED_STAGE" | cut -f1)"

# verify: distinct scene dirs and image-entry count == sum of source trees
n_scenes=$(tar --use-compress-program="zstd -d" -tf "$FUSED_STAGE" \
    | sed -n 's#^scans/\(scene[0-9]*_[0-9]*\)/.*#\1#p' | sort -u | wc -l)
n_full=$(tar --use-compress-program="zstd -d" -tf "$FUSED_STAGE" | grep -c '\.png$\|\.jpg$' || true)
n_src=$(find "$B1/scans" "$B2/scans" \( -name '*.png' -o -name '*.jpg' \) | wc -l)
echo "[fuse] distinct scenes in archive: $n_scenes (expected $((s1+s2)))"
echo "[fuse] image entries: archive=$n_full  source=$n_src"
if [ "$n_scenes" -ne "$((s1+s2))" ] || [ "$n_full" -ne "$n_src" ]; then
    echo "[fuse] ERROR: count mismatch"; exit 1
fi

echo "[fuse] copying single file to work ..."
cp "$FUSED_STAGE" "$FUSED_WORK.tmp" && mv "$FUSED_WORK.tmp" "$FUSED_WORK"
echo "[fuse] done -> $FUSED_WORK ($(du -h "$FUSED_WORK" | cut -f1))"
echo "[fuse] the two split tars are left in place; remove them once you adopt the fused tar."
