import os
from cloudmesh.common.console import Console
from chocolatechip import MySQLConnector
from moviepy.editor import VideoFileClip, CompositeVideoClip, ImageClip
from PIL import Image
import numpy as np
from yaspin import yaspin
from yaspin.spinners import Spinners

def process_video(mp4_file, conflict_coords):
    # Load the video file
    try:
        video = VideoFileClip(mp4_file)
    except OSError:
        Console.error(f"Error: Could not load video file {mp4_file}")
        return

    video_height = video.h  # Get the video height to invert Y coordinates
    video_width = video.w  # Get the video width

    # Create a red rectangle with transparency
    def make_rectangle(size, color=(255, 0, 0, 50)):  # RGBA where A is the alpha channel for transparency
        img = Image.new('RGBA', size, color)
        return np.array(img)

    # Define the size of the rectangle (width, height)
    rect_size = (100, 100)  # Change as needed

    # Create the rectangle image
    rect_img = make_rectangle(rect_size)

    print('listed as', conflict_coords)
    for x, y in conflict_coords:
        print(video_height - y, x)
        
    # Position the rectangles based on the new formula
    rect_clips = [
        ImageClip(rect_img, transparent=True).set_duration(video.duration)
        .set_position((y - rect_size[0] // 2, x - rect_size[1] // 2))
        for x, y in conflict_coords
    ]

    # Composite the rectangles onto the video
    final_video = CompositeVideoClip([video, *rect_clips])

    # Write the result to a file
    output_file = f"output_{mp4_file}"
    final_video.write_videofile(output_file, codec='libx264')

def main():
    # Check if there's at least one .mp4 file in the current directory
    if any(file.endswith('.mp4') for file in os.listdir('.')):
        print('ok')
    else:
        Console.error('i dont see any conflict videos')
        quit(1)

    mp4_files = [file for file in os.listdir('.') if file.endswith('.mp4')]

    with yaspin(Spinners.arc, text="Processing videos") as spinner:
        for index, mp4_file in enumerate(mp4_files):
            scratch = mp4_file.split('.mp4')[0]
            id1 = scratch.split('_')[4]
            id2 = scratch.split('_')[5]
            print(id1, id2)

            sql = MySQLConnector.MySQLConnector()
            # id_tuples is a list of tuples. Each tuple contains two IDs.
            df = sql.query_ttctable([(id1, id2)])
            print(df)
            print(df.columns)

            # get all the conflict_x and conflict_y coordinates
            conflict_coords = list(zip(df['conflict_x'], df['conflict_y']))
            print(conflict_coords)

            # Process the video with rectangles at all conflict points
            process_video(mp4_file, conflict_coords)

            # Update spinner text with remaining files count
            remaining = len(mp4_files) - (index + 1)
            spinner.text = f"Processing videos... {remaining} left to go"

        spinner.ok("✔ All videos processed!")

if __name__ == "__main__":
    main()
