#!/bin/bash
# Pack the scratch-built instance dataset into ONE zstd-compressed tar and
# place it in the work folder (work is fine with big single files, bad with
# many small ones). Training jobs then copy this single file to the node's
# local $TMPDIR and untar it there for fast I/O.
#
# Layout inside the archive:  scans/<scene>/raw_data/{subset,masks,masks_instance}/...
set -euo pipefail

BUILD=/cluster/scratch/niacobone/scannet_build          # contains scans/
STAGE_TAR=/cluster/scratch/niacobone/scannet_instance_dataset.tar.zst
WORK_DIR=/cluster/work/igp_psr/niacobone/distillation/dataset/scannet
WORK_TAR="$WORK_DIR/scannet_instance_dataset.tar.zst"

echo "[pack] building archive on scratch ..."
# zstd -1 -T0: light, multithreaded; all-zero masks compress very well.
tar --use-compress-program="zstd -1 -T0" \
    -C "$BUILD" -cf "$STAGE_TAR" scans
echo "[pack] archive size: $(du -h "$STAGE_TAR" | cut -f1)"

echo "[pack] verifying archive lists without error ..."
n_in_tar=$(tar --use-compress-program="zstd -d" -tf "$STAGE_TAR" | grep -c '\.png$\|\.jpg$' || true)
echo "[pack] image entries in archive: $n_in_tar"

echo "[pack] copying single file to work ..."
cp "$STAGE_TAR" "$WORK_TAR.tmp"
mv "$WORK_TAR.tmp" "$WORK_TAR"
echo "[pack] done -> $WORK_TAR ($(du -h "$WORK_TAR" | cut -f1))"
