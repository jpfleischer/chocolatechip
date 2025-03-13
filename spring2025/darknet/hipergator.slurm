#!/bin/bash
#SBATCH --job-name=darknet_train
#SBATCH --output=darknet_train.out
#SBATCH --error=darknet_train.err
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1          # Change to your desired GPU type and count (e.g., geforce:2)
#SBATCH --time=03:00:00         # Adjust the time limit as needed
#SBATCH --mem=50gb              # Adjust memory allocation



# Optionally, if you need to activate a virtual environment or set PATHs, do that here

# Run the container using Apptainer. Here we bind mount both your workspace (for training output)
# and your dataset directory so that your training data (in docker/) is available inside the container.
apptainer run --nv --fakeroot -B $(pwd)/darknet_workspace:/host_workspace -B $(pwd)/docker:/workspace my_darknet_container.sif
