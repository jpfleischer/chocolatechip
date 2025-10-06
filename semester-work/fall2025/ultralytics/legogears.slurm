#!/bin/bash
#SBATCH --job-name=lego_ultra
#SBATCH --output=%u_%x_%j.out
#SBATCH --error=%u_%x_%j.out
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --mem=32gb

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Load Apptainer if your site uses modules
if [ -f /etc/profile.d/modules.sh ]; then
  . /etc/profile.d/modules.sh
  module -q avail apptainer 2>/dev/null && module load apptainer || true
fi

mkdir -p outputs
nvidia-smi || true  # optional: log GPU info

apptainer run --nv \
  --bind "$PWD/outputs:/ultralytics/outputs" \
  my_ultralytics_container.sif
