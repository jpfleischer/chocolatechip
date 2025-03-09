#!/usr/bin/env python3
import os
import sys
from cloudmesh.common.console import Console
from cloudmesh.common.Shell import Shell
from docopt import docopt

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

    # Create the obj.names file with the specified contents
    obj_names_path = os.path.join(output_dir, "obj.names")
    names_content = "motorbike\ncar\ntruck\nbus\npedestrian\n"
    with open(obj_names_path, "w") as f:
        f.write(names_content)
    Console.ok(f"Created {obj_names_path} with object names.")

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
