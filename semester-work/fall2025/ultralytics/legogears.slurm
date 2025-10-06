#!/bin/bash
#SBATCH --job-name=lego_train
#SBATCH --output=%u_%x_%j.out
#SBATCH --error=%u_%x_%j.out
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --mem=32gb              # Adjust memory allocatio