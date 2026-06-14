#!/bin/bash
# Remove color/ frame dirs and leftover .sens from the work scans tree.
# These are not used downstream (subset/ is already derived; masks live elsewhere).
# Leaves subset/ masks/ masks_instance/ untouched.
set -e
SC=/cluster/work/igp_psr/niacobone/distillation/dataset/scannet/scans
n=0
for d in $(find "$SC" -mindepth 3 -maxdepth 3 -type d -name color); do
    rm -rf "$d"; n=$((n+1)); echo "[$n] removed $d"
done
for f in $(find "$SC" -maxdepth 2 -name "*.sens"); do
    rm -f "$f"; echo "removed $f"
done
echo "Prune done: $n color dirs removed."
