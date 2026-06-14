#!/bin/bash
# Bulk per-instance SAM3 masks for all 97 ScanNet scenes, in downstream
# priority order: val scenes (0080-0089) first, then 0000-0049, then 0050-0096.
# Resumable: scenes with masks_instance/.complete are skipped.
#
#SBATCH --job-name=instance_masks
#SBATCH --output=instance_masks_%j.log
#SBATCH --error=instance_masks_%j.err
#SBATCH --open-mode=append
#SBATCH --time=14:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4096
#SBATCH --gpus=rtx_4090:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=niacobone@student.ethz.ch

module purge
module load stack/2024-06 python/3.12.8 cuda/12.8.0 eth_proxy
cd /cluster/scratch/niacobone/sam3
source myenv/bin/activate

SCENES=()
for i in $(seq 80 89); do SCENES+=("$(printf "scene%04d_00" $i)"); done
for i in $(seq 0 49); do SCENES+=("$(printf "scene%04d_00" $i)"); done
for i in $(seq 50 79); do SCENES+=("$(printf "scene%04d_00" $i)"); done
for i in $(seq 90 96); do SCENES+=("$(printf "scene%04d_00" $i)"); done

mkdir -p logs
# Build into the fast scratch staging tree (subset/ + masks/ pre-staged there);
# masks_instance/ is written next to them. Resumable via per-scene .complete.
python scripts/save_instance_masks.py "${SCENES[@]}" \
    --scans_root /cluster/scratch/niacobone/scannet_build/scans \
    --log_file logs/instance_masks_bulk.jsonl

echo "Bulk instance-mask run finished."
