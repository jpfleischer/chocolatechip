import subprocess
import time
import xml.etree.ElementTree as ET
from matplotlib.ticker import FuncFormatter
from pprint import pprint
import curses
import asciichartpy
import os
from cloudmesh.common.console import Console
from cloudmesh.common.Shell import Shell
import pandas as pd
import matplotlib.pyplot as plt
from chocolatechip import MySQLConnector
import yaspin
from datetime import datetime

def docker_checker():
    """
    This function checks if the fastmot docker container is up.
    """
    # wait for fastmot to start. by using this command and querying.
    docker_up_command = 'docker ps -aqf status=running -f ancestor=fastmot-image | head -n 1'
    
    if subprocess.check_output(docker_up_command, shell=True).decode('utf-8').strip() != '':
        # fastmot is up
        return True
    else:
        return False
        

def fastmot_launcher(vid, vid2) -> str:
    """
    This function launches fastmot with the given video files.
    It returns the resolution of the videos.
    """
    path_to_fastmot = '/mnt/hdd/pipeline/fastmot'

    resolution_command = 'ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 '

    first_one = None
    for video in [vid, vid2]:
        # get resolution
        resolution = subprocess.check_output(resolution_command + video, shell=True).decode('utf-8').strip()
        if not first_one:
            first_one = resolution
        else:
            if first_one != resolution:
                Console.error('Resolutions are not the same. something is very wrong..')
                os._exit(1)
        
    # start fastmot
    # cd /mnt/hdd/pipeline/fastmot ; make VID="/mnt/hdd/gvideo/24_2024-03-04_18-50-00.000.mp4"
    fastmot_command = 'cd ' + path_to_fastmot + ' ; make dual-test VID=' + vid + ' VID2=' + vid2 + ' CAM_ID=' + vid.split('/')[-1].split('_')[0] + ' CAM_ID2=' + vid2.split('/')[-1].split('_')[0]
    print(fastmot_command)
    # time.sleep(3)
    subprocess.run(fastmot_command, shell=True, 
                    # get rid of output
                #    stdout=subprocess.DEVNULL,
                #    stderr=subprocess.DEVNULL,
    )

    print('going to wait for docker to come up')

    while True:
        if docker_checker():
            print('its up')
            print(first_one)
            return first_one
        else:
            time.sleep(1)
            print('.', end='', flush=True)
            continue


def nvidia_scraper() -> list:
    """
    This function uses the nvidia-smi command-line utility to get
    information about the GPUs in the system. It returns a list of
    dictionaries, where each dictionary contains the information
    for a single GPU.
    """
    result = subprocess.run(['nvidia-smi', '-x', '-q'], capture_output=True, text=True)

    # Parse the XML output
    root = ET.fromstring(result.stdout)

    info_list = []

    # Extract and print the desired information
    for gpu in root.iter('gpu'):
        id_of_gpu = gpu.get('id')
        name_of_gpu = gpu.find('product_name')
        memory_usage = gpu.find('fb_memory_usage/used')
        wattage = gpu.find('gpu_power_readings/power_draw')
        temperature = gpu.find('temperature/gpu_temp')
        fan_speed = gpu.find('fan_speed')
    
        info_dict = {
            'id': id_of_gpu,
            'name': name_of_gpu.text if name_of_gpu is not None else "N/A",
            'memory_usage': memory_usage.text if memory_usage is not None else "N/A",
            'wattage': wattage.text if wattage is not None else "N/A",
            'temperature': temperature.text if temperature is not None else "N/A",
            'fan_speed': fan_speed.text if fan_speed is not None else "N/A"
        }
        info_list.append(info_dict)

    return info_list


def curses_shower(stdscr):
    """
    This function is a curses application that prints the information
    for each GPU to the console. It refreshes every second.
    """
    while True:
        # Call the nvidia_scraper function
        info_list = nvidia_scraper()

        stdscr.clear()

        # Print the information for each GPU
        for info_dict in info_list:
            for key, value in info_dict.items():
                stdscr.addstr(f'{key}: {value}\n')
            stdscr.addstr('\n')

        stdscr.refresh()

        # Wait for 1 second
        time.sleep(1)


def dataframes_returner(res: str) -> list:
    """
    This function returns two DataFrames, one for each GPU. 
    So the first one in the list is the first gpu
    and the second one in the list is the second gpu. 
    It also prints the information to the console.

    res: str
        The resolution of the video. just for the print statement,
        not really vital.
    """
    data_list = [pd.DataFrame(), pd.DataFrame()]  # Initialize empty DataFrames
    start_time = datetime.now()  # Get the start time

    while True:
        if not docker_checker():
            break

        # Call the nvidia_scraper function
        info_list = nvidia_scraper()

        # Get the current time and calculate elapsed seconds
        timestamp = datetime.now()
        elapsed_seconds = (timestamp - start_time).total_seconds()

        # Update each DataFrame
        for i, info_dict in enumerate(info_list):
            new_row = pd.DataFrame(info_dict, index=[0])
            new_row['memory_usage'] = new_row['memory_usage'].str.replace(' MiB', '').astype(int)  # Convert memory_usage to int
            new_row['wattage'] = new_row['wattage'].str.replace(' W', '').astype(float)  # Convert wattage to float
            new_row['temperature'] = new_row['temperature'].str.replace('C', '').astype(int)  # Convert temperature to int
            new_row['fan_speed'] = new_row['fan_speed'].str.replace(' %', '').astype(int)  # Convert fan_speed to int

            # Add the elapsed seconds to the new row
            new_row['elapsed_seconds'] = elapsed_seconds

            data_list[i] = pd.concat([data_list[i], new_row])  # Add the new row to the DataFrame

            # clear terminal
            print(chr(27) + "[2J")

            for var in ['memory_usage', 'wattage', 'temperature', 'fan_speed']:
                # Print ASCII chart to console
                var_title = var.replace('_', ' ').title()
                the_best_title = f'{var_title} - {info_dict["name"]} #{i}'
                print(the_best_title)

                # Get all the data points
                values = data_list[i][var].values.tolist()
                print(asciichartpy.plot(values, {'height': 6}))
            print(res)

        time.sleep(1)  # Wait for 1 second

    return data_list

def gpu_plotter(info_list: list):
    for var in ['memory_usage', 'wattage', 'temperature', 'fan_speed']:
        for i, df in enumerate(info_list):
            fig, ax = plt.subplots()

            # Group by resolution and plot each group
            for resolution, group_df in df.groupby('resolution'):
                group_df.set_index('elapsed_seconds')[var].plot(ax=ax, label=f'Resolution {resolution}')

            var_title = var.replace('_', ' ').title()
            ax.set_title(f'{var_title} - GPU #{i}')

            # Add units to y-ticks
            if var == 'memory_usage':
                ax.set_ylabel('Memory Usage (MiB)')
                ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y)} MiB'))
            elif var == 'wattage':
                ax.set_ylabel('Wattage (W)')
                ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y)} W'))
            elif var == 'temperature':
                ax.set_ylabel('Temperature (C)')
                ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y)} C'))
            elif var == 'fan_speed':
                ax.set_ylabel('Fan Speed (%)')
                ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y)} %'))

            # Add 's' to x-ticks
            ax.set_xlabel('Elapsed Time (s)')
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)} s'))

            ax.grid(True)  # Add grid
            ax.legend()  # Add legend

            plt.savefig(f'gpu_{i}_{var}.png', bbox_inches='tight')  # Save the plot to an image file

def bar_plotter(tracks_with_res: dict,
                name: str = 'tracks_by_resolution.png'):
    # Create a new figure
    plt.figure()

    # Convert values to integers
    tracks_with_res = {k: int(v) for k, v in tracks_with_res.items()}

    # Create a bar plot
    plt.bar(tracks_with_res.keys(), tracks_with_res.values(), color='green')

    # Add title and labels
    plt.title('Number of Tracks by Resolution')
    plt.xlabel('Resolution')
    plt.ylabel('Number of Tracks')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Save the plot to an image file
    plt.savefig(name, bbox_inches='tight')


def main():
    vid1_list = [
        # "/mnt/hdd/gvideo/21_2023-08-30_19-45-04.000-medium.mp4",
        # "/mnt/hdd/gvideo/21_2023-08-30_19-45-04.000-medium-640.mp4",
        # "/mnt/hdd/gvideo/21_2023-08-30_19-45-04.000-medium-320.mp4"
        "/mnt/hdd/gvideo/25_2023-08-30_07-45-02.000-med-conflict.mp4",
        "/mnt/hdd/gvideo/25_2023-08-30_07-45-02.000-med-conflict-640.mp4",
        "/mnt/hdd/gvideo/25_2023-08-30_07-45-02.000-med-conflict-320.mp4"
    ]
    vid2_list = [
        # "/mnt/hdd/gvideo/22_2023-08-30_19-45-04.000-medium.mp4",
        # "/mnt/hdd/gvideo/22_2023-08-30_19-45-04.000-medium-640.mp4",
        # "/mnt/hdd/gvideo/22_2023-08-30_19-45-04.000-medium-320.mp4"
        "/mnt/hdd/gvideo/26_2023-08-30_07-45-02.000-med-conflict.mp4",
        "/mnt/hdd/gvideo/26_2023-08-30_07-45-02.000-med-conflict-640.mp4",
        "/mnt/hdd/gvideo/26_2023-08-30_07-45-02.000-med-conflict-320.mp4"
    ]
    # Define the parameters
    params = {
        'start_date': datetime.now(),  # Current time
        'end_date': datetime.now(),  # Current time
        'intersec_id': 3248,
        'cam_id': 27,  # Replace with the actual camera id
        'p2v': 1
    }

    mega_dfs = [pd.DataFrame(), pd.DataFrame()]
    tracks_with_res = {}
    tracks_df = {}
    conflict_df = {}

    sql = MySQLConnector.MySQLConnector()

    for i, (vid, vid2) in enumerate(zip(vid1_list, vid2_list)):
        # we do not want the comparison to be using previous tracks
        params['start_date'] = datetime.now()

        print('Starting the fastmot launcher')
        res = fastmot_launcher(vid, vid2)

        df_lists = dataframes_returner(res)

        # how many tracks?
        r = Shell.run("docker logs `docker ps -aqf ancestor=fastmot-image | head -n 1`")
        r2 = Shell.run("docker logs `docker ps -aqf ancestor=fastmot-image | sed -n '2p'`")

        list_of_logs = r.splitlines()
        list_of_logs2 = r2.splitlines()
        for line in list_of_logs:
            if 'Total number of tracks' in line:
                print('!!!!!!!')
                print(line)
                tracks_with_res[f"{res}-{i}-vid1"] = line.split(':')[-1].strip()
        for line in list_of_logs2:
            if 'Total number of tracks' in line:
                print('!!!!!!!')
                print(line)
                tracks_with_res[f"{res}-{i}-vid2"] = line.split(':')[-1].strip()

        # to all rows, add a column that says resolution, 
        # with value res
        for i in range(2):
            df_lists[i]['resolution'] = res
        mega_dfs[0] = pd.concat([mega_dfs[0], df_lists[0]])
        mega_dfs[1] = pd.concat([mega_dfs[1], df_lists[1]])
        print(mega_dfs[0].to_string())

        
        with yaspin.yaspin() as sp:
            sp.text = 'going to wait for the pipeline to finish'
            time.sleep(80)


        params['end_date'] = datetime.now()
        trackdf = sql.handleRequest(params, 'track')
        conflictdf = sql.handleRequest(params, 'conflict')
        tracks_df[f"{res}-{i}-vid1"] = trackdf
        conflict_df[f"{res}-{i}-vid1"] = conflictdf
        


    print('Starting gpu plotter')
    gpu_plotter(mega_dfs)

    pprint(tracks_with_res)
    print('Starting bar plotter')
    bar_plotter(tracks_with_res)

    # plain is a plain dictionary with names of runs attached
    # to how many tracks they generated.
    plain = {}

    for name, df in tracks_df.items():
        print(name, len(df))
        plain[name] = len(df)
    bar_plotter(plain, 'fancylen.png')
        # plot bar chart


    print('-'*20)
    print('-'*20)
    print('-'*20)
    for key, df in tracks_df.items():
        print(f"{key}:\n{df.to_string()}\n")
    pprint(conflict_df)
    print('Done')