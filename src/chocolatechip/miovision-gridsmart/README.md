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