# Video File Renaming Script

This script processes video files in a specified source directory, renaming them based on a predefined format. Files are either copied to a destination directory or, if the `--dryrun` flag is provided, the intended actions are printed without performing any file operations.

## Features

- **File Type Filtering**: Only `.mp4` files are processed; others are skipped.
- **Filename Parsing**: Automatically detects different filename formats (e.g., Gridsmart, Miovision) and renames them according to specified rules.
- **Dry Run Mode**: Simulate the renaming process to see intended actions without copying files.

## Requirements

- Python 3.x

## Usage

### Basic Command

Run the script with the following syntax:

```bash
sudo /home/jpf/ENV3/bin/python filemuncher.py /mnt/huge/DavieStirling /mnt/hdd/data/video_pipeline/
# or
sudo /home/jpf/ENV3/bin/python filemuncher.py /mnt/huge/DavieStirling/86dfd66a-0a38-4a4b-85ef-1f8fcdd3caa6-Sep_02_2024_18_00_18_15.mp4 /mnt/hdd/data/video_pipeline/
# or
sudo /home/jpf/ENV3/bin/python filemuncher.py /mnt/huge/66After /mnt/hdd/data/video_pipeline
```


```bash
# bring over
rclone copy onedrive:/BrowardVideos .

# when processing done
 5804  rclone check onedrive:/BrowardVideos . --include "*10-30*"
 5805  rclone check onedrive:/BrowardVideos . --include "*10-31*"
 # 2024/11/17 15:06:03 NOTICE: Local file system at /mnt/huge/66After: 0 differences found
 # 2024/11/17 15:06:03 NOTICE: Local file system at /mnt/huge/66After: 32 matching files
 5806  history
 5807  rclone delete onedrive:/BrowardVideos --include "*10-31*"
 5808  ls *10-31* | wc -l
 5809  ls *11-01* | wc -l
 5810  rclone check onedrive:/BrowardVideos . --include "*11-01*"
 5811  rclone delete onedrive:/BrowardVideos --include "*11-01*"
 5812  ls
 5813  ls *11-02* | wc -l
 5814  rclone check onedrive:/BrowardVideos . --include "*11-02*"
 5815  rclone delete onedrive:/BrowardVideos --include "*11-02*"
```

In the scenario that processing was stopped prematurely and messages purged

```bash
 sudo /home/jpf/ENV3/bin/python /home/jpf/chocolatechip/src/chocolatechip/miovision-gridsmart/filemuncher.py /mnt/huge/DavieStirling/hpg_resized /mnt/hdd/data/video_pipeline/tracking --check_whats_missing
```


Davie Road only.

```bash
# im on hipergator
python -m venv ~/ENV3
source ~/ENV3/bin/activate
pip install yaspin
# the slurm gpu thing did not work but doesnt matter
#srun --pty --qos=ranka --partition=gpu --gres=gpu:a100:1 --mem=32G --time=0:30:00 bash
module load ffmpeg
cd /blue/ranka/j.fleischer/DavieStirling
python padder.py --source original/ --destination resized/


#test
# singularity run --nv ffmpeg.sif \
#     -hwaccel cuda \
#     -i ../original/long-miovision-hex-string-Aug_05_2024_15_00_15_15.mp4 \
#     -vf "scale_cuda=960:960, pad=1280:960:160:0:black" \
#     -c:v h264_nvenc \
#     -preset fast \
#     output.mp4
```

```bash
 sudo /home/jpf/ENV3/bin/python /home/jpf/chocolatechip/ephemeral.py
```