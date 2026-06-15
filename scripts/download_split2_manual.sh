#!/bin/bash
# MANUAL split-2 .sens downloader -- run this yourself in a terminal on a LOGIN
# node (NOT via sbatch). Purpose: test whether kaldir is reachable outside of a
# SLURM compute node. Resumable: re-running only fetches what's missing.
#
#   bash scripts/download_split2_manual.sh            # scenes 97..199
#   bash scripts/download_split2_manual.sh 97 110     # custom range
#
# Watch the output:
#   "[sceneXXXX_00] OK N.NN GB ..."  -> server reachable, download working.
#   "... TimeoutError / failed"      -> still unreachable from here too.

START=${1:-97}
END=${2:-199}
BUILD2=/cluster/scratch/niacobone/scannet_build_split2
PY=/cluster/scratch/niacobone/sam3/myenv/bin/python
DL=/cluster/scratch/niacobone/sam3/scripts/download_sens.py

# Load the ETH proxy if the module system is available (harmless on login nodes
# that already have direct network).
module load eth_proxy 2>/dev/null || module load stack/2024-06 eth_proxy 2>/dev/null || true

echo "=== quick reachability probe (first scene, 60s) ==="
"$PY" "$DL" -o "$BUILD2" --start "$START" --end "$START" --timeout 30 --retries 2

echo
echo "=== full range $START..$END ==="
"$PY" "$DL" -o "$BUILD2" --start "$START" --end "$END" --timeout 180 --retries 5

echo
echo "Downloaded .sens so far:"
find "$BUILD2/scans" -name '*.sens' 2>/dev/null | wc -l
