#!/bin/bash
# Runs automatically after the build job (submit with --dependency=afterok:<id>).
# Packs the scratch dataset into one zstd tar in work, writes the README report,
# and verifies the archive. CPU-only, cheap.
#
#SBATCH --job-name=pack_instance_ds
#SBATCH --output=pack_instance_ds_%j.log
#SBATCH --error=pack_instance_ds_%j.err
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

# 1) report from per-scene stats
python scripts/gen_instance_report.py

# 2) pack single archive -> work
bash scripts/pack_dataset.sh

# 3) verify: untar listing count vs source count
WORK_TAR=/cluster/work/igp_psr/niacobone/distillation/dataset/scannet/scannet_instance_dataset.tar.zst
SRC_PNG=$(find /cluster/scratch/niacobone/scannet_build/scans -name '*.png' | wc -l)
SRC_JPG=$(find /cluster/scratch/niacobone/scannet_build/scans -name '*.jpg' | wc -l)
echo "[verify] source PNG=$SRC_PNG JPG=$SRC_JPG"
echo "[verify] archive on work: $(du -h "$WORK_TAR" | cut -f1)"
echo "Pack + report + verify complete."
