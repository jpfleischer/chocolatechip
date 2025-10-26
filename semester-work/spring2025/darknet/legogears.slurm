#!/bin/bash
#SBATCH --job-name=lego_train
#SBATCH --output=%u_%x_%j.out
#SBATCH --error=%u_%x_%j.out
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --mem=32gb              # Adjust memory allocation
#SBATCH --partition=hpg-b200 


# Optionally, if you need to activate a virtual environment or set PATHs, do that here

# Run the container using Apptainer. Here we bind mount both your workspace (for training output)
# and your dataset directory so that your training data (in docker/) is available inside the container.


set -euo pipefail
# Ensure we run from the submit directory (important on some clusters)
cd "${SLURM_SUBMIT_DIR:-$PWD}"
make arun
