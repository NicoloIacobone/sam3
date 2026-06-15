#!/bin/bash
# Self-healing .sens downloader for split-2 (scene0097_00..scene0199_00).
# Retries the whole range every RETRY_SLEEP seconds until every .sens is present
# (kaldir is currently down). Exits 0 only when ALL are present, so an
# afterok-dependent build job fires exactly when the data is complete. If the
# wall-time runs out before the server returns, resubmit this script.
#
#SBATCH --job-name=split2_dl_selfheal
#SBATCH --output=split2_dl_selfheal_%j.log
#SBATCH --error=split2_dl_selfheal_%j.err
#SBATCH --open-mode=append
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4096
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=niacobone@student.ethz.ch

module load eth_proxy 2>/dev/null || module load stack/2024-06 eth_proxy 2>/dev/null || true
set -u

START=${1:-97}
END=${2:-199}
RETRY_SLEEP=${3:-1800}      # 30 min between attempts
BUILD2=/cluster/scratch/niacobone/scannet_build_split2
PY=/cluster/scratch/niacobone/sam3/myenv/bin/python
DL=/cluster/scratch/niacobone/sam3/scripts/download_sens.py
mkdir -p "$BUILD2/scans"

count_missing() {
    local n=0 i S
    for i in $(seq "$START" "$END"); do
        S=$(printf "scene%04d_00" "$i")
        [ -s "$BUILD2/scans/$S/$S.sens" ] || n=$((n+1))
    done
    echo "$n"
}

attempt=0
while true; do
    attempt=$((attempt+1))
    miss=$(count_missing)
    echo "=== $(date '+%F %T') attempt $attempt | missing=$miss/$((END-START+1)) ==="
    if [ "$miss" -eq 0 ]; then
        echo "ALL .sens PRESENT -> exiting 0 so the build can start."
        exit 0
    fi
    # one resumable pass (fails fast per scene if server still down)
    "$PY" "$DL" -o "$BUILD2" --start "$START" --end "$END" --timeout 180 --retries 2 || true
    miss=$(count_missing)
    if [ "$miss" -eq 0 ]; then
        echo "ALL .sens PRESENT after attempt $attempt -> exiting 0."
        exit 0
    fi
    echo "still missing $miss; sleeping ${RETRY_SLEEP}s ..."
    sleep "$RETRY_SLEEP"
done
