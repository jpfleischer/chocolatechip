
#!/bin/bash
#SBATCH --job-name=lego_train
#SBATCH --output=%u_%x_%j.out
#SBATCH --error=%u_%x_%j.out
#SBATCH --partition=gpu
#SBATCH --gpus=v100:1          # Change to your desired GPU type and count (e.g., geforce:2)
#SBATCH --time=05:00:00
#SBATCH --mem=50gb              # Adjust memory allocation



# Optionally, if you need to activate a virtual environment or set PATHs, do that here

# Run the container using Apptainer. Here we bind mount both your workspace (for training output)
# and your dataset directory so that your training data (in docker/) is available inside the container.


apptainer exec --nv \
    my_darknet_container.sif cms gpu watch --gpu=$(echo $CUDA_VISIBLE_DEVICES | tr -d '\n\r') --delay=1 --dense --logfile=mylogfile.log &
# sleep 25

make arun-command

pkill -SIGINT -f "cms gpu watch"

sleep 5

# Move the mylogfile.log to the most recently created directory in LegoGearsFiles/outputs/
target=$(ls -td LegoGearsFiles/outputs/*/ | head -1)
mv mylogfile.log "$target"
