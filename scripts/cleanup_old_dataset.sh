#!/bin/bash
# Consolidate the work dataset dir down to just the full tar (+ tsv + READMEs).
# Safety gate: re-verify the full tar's integrity BEFORE deleting anything.
# Deletes: the two split tars and the scattered per-scene scans/ directory.
# Keeps:   scannet_instance_dataset_full.tar.zst, scannetv2-labels.combined.tsv,
#          INSTANCE_MASKS_README.md, INSTANCE_MASKS_README_split2.md
#
#SBATCH --job-name=cleanup_old_dataset
#SBATCH --output=cleanup_old_dataset_%j.log
#SBATCH --error=cleanup_old_dataset_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4096
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=niacobone@student.ethz.ch

set -euo pipefail
W=/cluster/work/igp_psr/niacobone/distillation/dataset/scannet
FULL="$W/scannet_instance_dataset_full.tar.zst"

echo "=== safety gate: verifying $FULL ==="
if ! zstd -t "$FULL"; then
    echo "ABORT: full tar failed integrity check; deleting nothing."
    exit 1
fi
n=$(tar --use-compress-program="zstd -d" -tf "$FULL" \
    | sed -n 's#^scans/\(scene[0-9]*_[0-9]*\)/.*#\1#p' | sort -u | wc -l)
echo "full tar OK, distinct scenes=$n"
if [ "$n" -ne 200 ]; then echo "ABORT: expected 200 scenes, got $n"; exit 1; fi

echo "=== deleting split tars ==="
rm -fv "$W/scannet_instance_dataset.tar.zst" "$W/scannet_instance_dataset_split2.tar.zst"

echo "=== deleting scattered scans/ ($(find "$W/scans" -type f 2>/dev/null | wc -l) files) ==="
rm -rf "$W/scans"

echo "=== remaining contents ==="
ls -lh "$W"
echo "cleanup complete."
