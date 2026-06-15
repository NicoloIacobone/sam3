#!/bin/bash
# Pack split-2 into one zstd tar in work + write its report. Submit with
# --dependency=afterok:<process_split2 job id>. CPU-only.
#
#SBATCH --job-name=pack_split2
#SBATCH --output=pack_split2_%j.log
#SBATCH --error=pack_split2_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4096
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=niacobone@student.ethz.ch

module purge
module load stack/2024-06 python/3.12.8 eth_proxy
cd /cluster/scratch/niacobone/sam3
source myenv/bin/activate
set -e

BUILD2=/cluster/scratch/niacobone/scannet_build_split2
WORK=/cluster/work/igp_psr/niacobone/distillation/dataset/scannet
STAGE_TAR=/cluster/scratch/niacobone/scannet_instance_dataset_split2.tar.zst
WORK_TAR="$WORK/scannet_instance_dataset_split2.tar.zst"

# report for split-2 scenes
python scripts/gen_instance_report.py --build "$BUILD2" \
    --out "$WORK/INSTANCE_MASKS_README_split2.md"

echo "[pack2] building archive ..."
tar --use-compress-program="zstd -1 -T0" -C "$BUILD2" -cf "$STAGE_TAR" scans
echo "[pack2] archive size: $(du -h "$STAGE_TAR" | cut -f1)"

n_tar=$(tar --use-compress-program="zstd -d" -tf "$STAGE_TAR" | grep -c '\.png$\|\.jpg$' || true)
n_src=$(find "$BUILD2/scans" \( -name '*.png' -o -name '*.jpg' \) | wc -l)
echo "[pack2] entries: archive=$n_tar source=$n_src"

cp "$STAGE_TAR" "$WORK_TAR.tmp" && mv "$WORK_TAR.tmp" "$WORK_TAR"
echo "[pack2] done -> $WORK_TAR ($(du -h "$WORK_TAR" | cut -f1))"
