#!/usr/bin/env python3
import os
import sys
import urllib.request
import shutil
from cloudmesh.common.console import Console
from cloudmesh.common.Shell import Shell
from docopt import docopt

def download_file(url, dest_path):
    """
    Download a file from the given URL to the destination path.
    Uses a custom User-Agent header and reads the file in chunks.
    """
    Console.info(f"Downloading weights file from {url} to {dest_path}")
    try:
        # Create a request with a custom User-Agent header.
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                out_file.write(chunk)
        Console.ok(f"Downloaded weights file to {dest_path}")
    except Exception as e:
        Console.error(f"Failed to download weights file: {e}")

def extract_frames(filename, timestamp, num_frames=30):
    """
    Extract frames from a video file using ffmpeg.
    
    Parameters:
        filename (str): Path to the video file.
        timestamp (str): Starting timestamp in HH:MM:SS (or MM:SS) format.
        num_frames (int): Number of frames to extract (default 30).
        
    The output directory is generated from the video file’s basename 
    (e.g. "video.mp4" yields "video_frames").
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    output_dir = f"{base}_frames"
    os.makedirs(output_dir, exist_ok=True)
    
    # Build the ffmpeg command without the redundant mkdir command.
    cmd = (
        f'ffmpeg -ss {timestamp} -i {filename} '
        f'-vf "select=\'lte(n\\,{num_frames-1})\'" -vsync vfr {output_dir}/frame_%02d.png'
    )
    Console.info(f"Running command: {cmd}")
    Shell.run(cmd)
    Console.ok(f"Extracted frames to {output_dir}")

    # Create the obj.names file with the specified contents.
    obj_names_path = os.path.join(output_dir, "obj.names")
    names_content = "motorbike\ncar\ntruck\nbus\npedestrian\n"
    with open(obj_names_path, "w") as f:
        f.write(names_content)
    Console.ok(f"Created {obj_names_path} with object names.")

    # Download the weights file into the output directory.
    weights_url = "https://www.dropbox.com/scl/fi/1cdwmirh7x1ocpjiu4a2x/annotations_best_3_9_25.weights?rlkey=wx7x5vyjjfebd6oohptlacta0&st=scvcucoe&dl=1"
    weights_path = os.path.join(output_dir, "annotation_3_9_25.weights")
    download_file(weights_url, weights_path)

    # Copy annotations.cfg from the script's directory to the output folder.
    script_dir = os.path.dirname(os.path.realpath(__file__))
    annotations_cfg_src = os.path.join(script_dir, "annotations.cfg")
    if os.path.exists(annotations_cfg_src):
        dest_cfg = os.path.join(output_dir, "annotations.cfg")
        try:
            shutil.copy(annotations_cfg_src, dest_cfg)
            Console.ok(f"Copied annotations.cfg to {dest_cfg}")
        except Exception as e:
            Console.error(f"Error copying annotations.cfg: {e}")
    else:
        Console.warn(f"annotations.cfg not found in {script_dir}")

def main():
    """
Usage:
  extract <filename> <timestamp> [<frames>]
    """
    # Only parse the arguments after "extract"
    args = docopt(main.__doc__, argv=sys.argv[2:])
    
    filename = args['<filename>']
    timestamp = args['<timestamp>']
    frames = args['<frames>'] if args['<frames>'] is not None else '30'
    
    try:
        num_frames = int(frames)
    except ValueError:
        Console.error("Frames must be an integer.")
        return

    extract_frames(filename, timestamp, num_frames)

if __name__ == '__main__':
    main()
