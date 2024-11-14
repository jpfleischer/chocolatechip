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
```