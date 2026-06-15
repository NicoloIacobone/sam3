#!/bin/bash
# Non-interactive download of ONLY the .sens files for the split-2 scene range
# (scene0097_00 .. scene0199_00) into the scratch split-2 build tree.
#
# The ScanNet downloader has two interactive input() prompts (TOS agreement and
# a "press n to skip .sens" prompt). Piping `yes ''` answers both with Enter.
# `--type .sens` fetches only the .sens (not all 13 file types). Resumable:
# scenes whose .sens already exists are skipped.
#
# Dual-use: `sbatch scripts/download_split2.sh [start] [end]` (robust, survives
# logout) or `bash scripts/download_split2.sh [start] [end]` on a login node.
#
#SBATCH --job-name=split2_download
#SBATCH --output=split2_download_%j.log
#SBATCH --error=split2_download_%j.err
#SBATCH --open-mode=append
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4096
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=niacobone@student.ethz.ch

# eth_proxy gives compute nodes outbound network access for the download.
module load eth_proxy 2>/dev/null || module load stack/2024-06 eth_proxy 2>/dev/null || true
set -u

START=${1:-97}
END=${2:-199}
BUILD2=/cluster/scratch/niacobone/scannet_build_split2
PY=/cluster/scratch/niacobone/sam3/myenv/bin/python
DL=/cluster/scratch/niacobone/sam3/scripts/download_sens.py

mkdir -p "$BUILD2"
# Robust direct .sens fetch (timeouts + retries + resumable). If the TUM server
# is down, scenes fail-fast (no hang) and a re-run resumes the missing ones.
"$PY" "$DL" -o "$BUILD2" --start "$START" --end "$END"
