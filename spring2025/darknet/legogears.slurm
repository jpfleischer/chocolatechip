#!/bin/bash
#SBATCH --job-name=lego_train
#SBATCH --output=%u_%x_%j.out
#SBATCH --error=%u_%x_%j.out
#SBATCH --partition=gpu
#SBATCH --gpus=geforce:1          # Change to your desired GPU type and count (e.g., geforce:2)
#SBATCH --time=03:00:00
#SBATCH --mem=32gb              # Adjust memory allocation



# Optionally, if you need to activate a virtual environment or set PATHs, do that here

# Run the container using Apptainer. Here we bind mount both your workspace (for training output)
# and your dataset directory so that your training data (in docker/) is available inside the container.


make lego-setup

make arun-cms-gpu
# sleep 25

make arun-command

pkill -SIGINT -f "cms gpu watch"

sleep 5

# Move the mylogfile.log to the most recently created directory in LegoGearsFiles/outputs/
target=$(ls -td LegoGearsFiles/outputs/*/ | head -1)
mv mylogfile.log "$target"
