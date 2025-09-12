#!/bin/bash
#SBATCH --job-name=darknet_train
#SBATCH --output=%u_%x_%j.out
#SBATCH --error=%u_%x_%j.out
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=a100:4          # Change to your desired GPU type and count (e.g., geforce:2)
#SBATCH --time=3-00:00:00
#SBATCH --mem=50gb              # Adjust memory allocation



# Optionally, if you need to activate a virtual environment or set PATHs, do that here

# Run the container using Apptainer. Here we bind mount both your workspace (for training output)
# and your dataset directory so that your training data (in docker/) is available inside the container.

cms gpu watch --gpu=$(echo $CUDA_VISIBLE_DEVICES | tr -d '\n\r') --delay=0.5 --dense > cars_gpu.log &

make arun-cars-command

target=$(ls -td CollabFiles/outputs/*/ | head -1)
mv cars_gpu.log "$target"
