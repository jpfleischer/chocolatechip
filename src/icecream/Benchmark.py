import subprocess
import time
import xml.etree.ElementTree as ET
from matplotlib.ticker import FuncFormatter

import curses
import datetime
import asciichartpy
import os
from cloudmesh.common.console import Console
import pandas as pd
import matplotlib.pyplot as plt


def docker_checker():    
    # wait for fastmot to start. by using this command and querying.
    docker_up_command = 'docker ps -aqf status=running -f ancestor=fastmot-image | head -n 1'
    
    if subprocess.check_output(docker_up_command, shell=True).decode('utf-8').strip() != '':
        # fastmot is up
        return True
    else:
        return False
        

def fastmot_launcher(vid, vid2) -> str:
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


def dataframes_returner() -> list:
    data_list = [pd.DataFrame(), pd.DataFrame()]  # Initialize empty DataFrames
    start_time = datetime.datetime.now()  # Get the start time

    while True:
        if not docker_checker():
            break

        # Call the nvidia_scraper function
        info_list = nvidia_scraper()

        # Get the current time and calculate elapsed seconds
        timestamp = datetime.datetime.now()
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

            print(chr(27) + "[2J")


            for var in ['memory_usage', 'wattage', 'temperature', 'fan_speed']:
                # Print ASCII chart to console
                var_title = var.replace('_', ' ').title()
                the_best_title = f'{var_title} - {info_dict["name"]} #{i}'
                print(the_best_title)
                print(asciichartpy.plot(data_list[i][var].values.tolist()))

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

if __name__ == "__main__":
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

    mega_dfs = [pd.DataFrame(), pd.DataFrame()]

    for vid, vid2 in zip(vid1_list, vid2_list):
        
        print('Starting the fastmot launcher')
        res = fastmot_launcher(vid, vid2)

        df_lists = dataframes_returner()
        # to all rows, add a column that says resolution, 
        # with value res
        for i in range(2):
            df_lists[i]['resolution'] = res
        mega_dfs[0] = pd.concat([mega_dfs[0], df_lists[0]])
        mega_dfs[1] = pd.concat([mega_dfs[1], df_lists[1]])
        print(mega_dfs[0].to_string())


    print('Starting gpu plotter')
    gpu_plotter(mega_dfs)

    print('Done')