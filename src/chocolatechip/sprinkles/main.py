import os
import requests
from cloudmesh.common.console import Console
from cloudmesh.common.systeminfo import os_is_windows
from cloudmesh.common.Shell import Shell
from cloudmesh.common.util import path_expand
from moviepy.config import change_settings
from chocolatechip import MySQLConnector
from moviepy.editor import VideoFileClip, CompositeVideoClip, ImageClip, TextClip
from PIL import Image, ImageDraw
import numpy as np
from yaspin import yaspin
from yaspin.spinners import Spinners

def process_video(mp4_file, conflict_coords, char_placement, thumbnail_size):
    # Load the video file
    try:
        video = VideoFileClip(mp4_file)
    except OSError:
        Console.error(f"Error: Could not load video file {mp4_file}")
        return

    video_height = video.h  # Get the video height
    video_width = video.w  # Get the video width

    if char_placement in ['28', '29']:
        picture = '30_Map.png'
    elif char_placement in ['21', '22']:
        picture = '21_Map.png'
    elif char_placement in ['31', '32']:
        picture = '33_Map.png'
    elif char_placement in ['25', '26']:
        picture = '27_Map.png'
    elif char_placement in ['34', '35']:
        picture = '34_Map.png'
    elif char_placement in ['24']:
        picture = '24_Map.png'

    # Load the map image
    map_img = Image.open(picture)
    draw = ImageDraw.Draw(map_img)

    # Define the size of the rectangle (width, height)
    rect_size = (100, 100)  # Change as needed

    # Draw the rectangles on the map image
    for x, y in conflict_coords:
        realx = x - rect_size[0] // 2
        realy = y - rect_size[1] // 2
        draw.rectangle([realx, realy, realx + rect_size[0], realy + rect_size[1]], fill=(255, 0, 0, 128), outline=(255, 0, 0, 128), width=3)

    # Convert the map image with rectangles to an image clip
    map_img_clip = ImageClip(np.array(map_img)).set_duration(video.duration).set_position((0, 0)).resize(thumbnail_size)

    # Create a red rectangle with transparency for conflict points on the video
    # def make_rectangle(size, color=(255, 0, 0, 50)):  # RGBA where A is the alpha channel for transparency
    #     img = Image.new('RGBA', size, color)
    #     return np.array(img)

    # # Create the rectangle image
    # rect_img = make_rectangle(rect_size)

    # # Position the rectangles on the video
    # rect_clips = [
    #     ImageClip(rect_img, transparent=True).set_duration(video.duration)
    #     .set_position((x - rect_size[0] // 2, video_height - y - rect_size[1] // 2))
    #     for x, y in conflict_coords
    # ]

    # Add character "W" and "E" based on char_placement
    text_clips = []
    if char_placement == "28":
        w_clip = TextClip("W", fontsize=70, color='white').set_duration(video.duration).set_position(("right", "bottom"))
        e_clip = TextClip("E", fontsize=70, color='white').set_duration(video.duration).set_position(("left", "top"))
        text_clips.extend([w_clip, e_clip])
    elif char_placement == "29":
        w_clip = TextClip("W", fontsize=70, color='white').set_duration(video.duration).set_position(("left", "top"))
        e_clip = TextClip("E", fontsize=70, color='white').set_duration(video.duration).set_position(("right", "bottom"))
        text_clips.extend([w_clip, e_clip])
    elif char_placement == "21":
        w_clip = TextClip("W", fontsize=70, color='white').set_duration(video.duration).set_position(("left", "top"))
        e_clip = TextClip("E", fontsize=70, color='white').set_duration(video.duration).set_position(("right", "bottom"))
        text_clips.extend([w_clip, e_clip])
    elif char_placement == "22":   
        w_clip = TextClip("N", fontsize=70, color='white').set_duration(video.duration).set_position(("left", "top"))
        e_clip = TextClip("S", fontsize=70, color='white').set_duration(video.duration).set_position(("right", "bottom"))
        text_clips.extend([w_clip, e_clip]) 
    elif char_placement == "24":
        nw_clip = TextClip("NW", fontsize=70, color='white').set_duration(video.duration).set_position(("left", "top"))
        se_clip = TextClip("SE", fontsize=70, color='white').set_duration(video.duration).set_position(("right", "bottom"))
        text_clips.extend([nw_clip, se_clip])
    elif char_placement == "25":
        w_clip = TextClip("E", fontsize=70, color='white').set_duration(video.duration).set_position(("left", "top"))
        e_clip = TextClip("W", fontsize=70, color='white').set_duration(video.duration).set_position(("right", "bottom"))
        text_clips.extend([w_clip, e_clip]) 
    elif char_placement == "26":
        w_clip = TextClip("W", fontsize=70, color='white').set_duration(video.duration).set_position(("left", "top"))
        e_clip = TextClip("E", fontsize=70, color='white').set_duration(video.duration).set_position(("right", "bottom"))
        text_clips.extend([w_clip, e_clip]) 
    elif char_placement == "31":
        w_clip = TextClip("E", fontsize=70, color='white').set_duration(video.duration).set_position(("left", "top"))
        e_clip = TextClip("N", fontsize=70, color='white').set_duration(video.duration).set_position(("right", "bottom"))
        text_clips.extend([w_clip, e_clip]) 
    elif char_placement == "32":
        w_clip = TextClip("N", fontsize=70, color='white').set_duration(video.duration).set_position(("left", "top"))
        e_clip = TextClip("S", fontsize=70, color='white').set_duration(video.duration).set_position(("right", "bottom"))
        text_clips.extend([w_clip, e_clip]) 
    elif char_placement == "34":
        ne_clip = TextClip("NE", fontsize=70, color='white').set_duration(video.duration).set_position(("left", "top"))
        sw_clip = TextClip("SW", fontsize=70, color='white').set_duration(video.duration).set_position(("right", "bottom"))
        text_clips.extend([ne_clip, sw_clip])
    elif char_placement == "35":
        nw_clip = TextClip("NW", fontsize=70, color='white').set_duration(video.duration).set_position(("left", "top"))
        se_clip = TextClip("SE", fontsize=70, color='white').set_duration(video.duration).set_position(("right", "bottom"))
        text_clips.extend([nw_clip, se_clip])
    

    # Composite the map image, rectangles, and text onto the video
    # final_video = CompositeVideoClip([video, map_img_clip, *rect_clips, *text_clips])
    final_video = CompositeVideoClip([video, map_img_clip, *text_clips])
# 
    # Write the result to a file
    output_file = f"output_{mp4_file}"
    full = path_expand(f"~/sprinkle_output/{output_file}")

    
    final_video.write_videofile(full, 
                                codec='libx264',
                                threads=8,)

def main():
    if os_is_windows():
        program_files = Shell.map_filename(r'C:\Program Files').path

        program_files_dirs = None
        for path, dirnames, filenames in os.walk(program_files):
            program_files_dirs = dirnames
            break

        imagemagick_dir = None
        if program_files_dirs:
            for dirname in program_files_dirs:
                if 'ImageMagick' in dirname:
                    imagemagick_dir = dirname
                    break

        if imagemagick_dir is None:
            raise FileNotFoundError('imagemagick not installed! install it from https://imagemagick.org/script/download.php#windows')
        else:
            installation = fr'C:\\Program Files\\{imagemagick_dir}\\magick.exe'
            print(installation)
            # IMAGEMAGICK_BINARY = os.getenv('IMAGEMAGICK_BINARY', installation)

            change_settings({"IMAGEMAGICK_BINARY": installation})

    # Check if there's at least one .mp4 file in the current directory
    if any(file.endswith('.mp4') for file in os.listdir('.')):
        print('ok')
    else:
        Console.error('i dont see any conflict videos')
        quit(1)

    for picture in ['30_Map.png', '21_Map.png', '33_Map.png', '27_Map.png', '34_Map.png', '24_Map.png']:
        if not os.path.isfile(picture):
            Console.info('Downloading photo')
            url = f'http://maltlab.cise.ufl.edu:30101/api/image/{picture}'
            response = requests.get(url)
            with open(picture, 'wb') as file:
                file.write(response.content)


    # home_dir_sprinkle = path_expand("~/sprinkle_output")
    # if not os.path.isdir(home_dir_sprinkle):
    #     Shell.mkdir(home_dir_sprinkle)

    mp4_files = [file for file in os.listdir('.') if file.endswith('.mp4')]

    thumbnail_size = (200, 150)  # Specify the size of the thumbnail (width, height)

    with yaspin(Spinners.arc, text="Processing videos") as spinner:
        for index, mp4_file in enumerate(mp4_files):
            scratch = mp4_file.split('.mp4')[0]
            id1 = scratch.split('_')[4]
            id2 = scratch.split('_')[5]
            print(id1, id2)

            char_placement = scratch.split('_')[1]

            sql = MySQLConnector.MySQLConnector()
            # id_tuples is a list of tuples. Each tuple contains two IDs.
            df = sql.query_ttctable([(id1, id2)])
            print(df)
            print(df.columns)

            # get all the conflict_x and conflict_y coordinates
            conflict_coords = list(zip(df['conflict_x'], df['conflict_y']))
            print(conflict_coords)

            # Process the video with rectangles at all conflict points and add characters
            process_video(mp4_file, conflict_coords, char_placement, thumbnail_size)

            # Update spinner text with remaining files count
            remaining = len(mp4_files) - (index + 1)
            spinner.text = f"Processing videos... {remaining} left to go"

        spinner.ok("✔ All videos processed!")

if __name__ == "__main__":
    main()
