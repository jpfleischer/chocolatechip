#!/bin/bash
#SBATCH --job-name=darknet_train
#SBATCH --output=%u_%x_%j.out
#SBATCH --error=%u_%x_%j.out
#SBATCH --partition=gpu
#SBATCH --gpus=geforce:1          # Change to your desired GPU type and count (e.g., geforce:2)
#SBATCH --time=01:00:00
#SBATCH --mem=50gb              # Adjust memory allocation



# Optionally, if you need to activate a virtual environment or set PATHs, do that here

# Run the container using Apptainer. Here we bind mount both your workspace (for training output)
# and your dataset directory so that your training data (in docker/) is available inside the container.
make arun-command
mv *.out /outputs

