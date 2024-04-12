import seaborn as sns
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
import json
from datetime import datetime
import docker

def docker_checker(number_wanted_running: int=1):
    """
    This function checks if the fastmot docker container is up.
    """
    # wait for fastmot to start. by using this command and querying.
    docker_up_command = 'docker ps -aqf status=running -f ancestor=fastmot-image | wc -l'
    
    output = int(subprocess.check_output(docker_up_command, shell=True).decode('utf-8').strip())

    if output == number_wanted_running:
        # fastmot is up
        return True
    print("i want", number_wanted_running, "and the output is", output)
    return False
        

def fastmot_launcher(vid, 
                     vid2,
                     streams: int=1,
                     rtsp: bool=True) -> str:
    """
    This function launches fastmot with the given video files.
    It returns the resolution of the videos.
    """

    # uncomment this for rtsp dual
    if rtsp:
        path_to_fastmot = '/mnt/hdd/pipeline/fastmot'
    else:
        path_to_fastmot = '/home/jpf/fastmot'


    resolution_command = 'ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 '

    first_resolution_found = None
    for video in [vid, vid2]:
        # get resolution
        resolution = subprocess.check_output(resolution_command + video, shell=True).decode('utf-8').strip()
        if not first_resolution_found:
            first_resolution_found = resolution
        else:
            if first_resolution_found != resolution:
                Console.error('Resolutions are not the same. something is very wrong..')
                os._exit(1)
        
    # start fastmot
    # cd /mnt/hdd/pipeline/fastmot ; make VID="/mnt/hdd/gvideo/24_2024-03-04_18-50-00.000.mp4"

    Shell.run(f'cd {path_to_fastmot} ; make down')

    if rtsp:
        for _ in range(streams):
            # fastmot_command = 'cd ' + path_to_fastmot + ' ; make dual-test VID=' + vid + ' VID2=' + vid2 + ' CAM_ID=' + vid.split('/')[-1].split('_')[0] + ' CAM_ID2=' + vid2.split('/')[-1].split('_')[0]
            fastmot_command = 'cd ' + path_to_fastmot + ' ; make stress-test VID=' + vid + ' CAM_ID=' + vid.split('/')[-1].split('_')[0]
            print(fastmot_command)
            # time.sleep(3)
            subprocess.run(fastmot_command, shell=True, 
                            # get rid of output
                        #    stdout=subprocess.DEVNULL,
                        #    stderr=subprocess.DEVNULL,
            )

    else:
        fastmot_command = 'cd ' + path_to_fastmot + '; make run'
                


    print('going to wait for docker to come up')

    while True:
        if docker_checker(streams):
            print('its up')
            print(first_resolution_found)
            return first_resolution_found
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

def docker_stats_grabber(container_name: str) -> list:
    """
    dataframe for each container
    """
    return_list = []
    cmd = f"docker ps -aqf status=running -f ancestor=fastmot-image"
    results = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    for docker_id in results.stdout.splitlines():
        docker_stats_cmd = f'docker stats --no-stream {docker_id} --format "{{{{ json . }}}}"'
        
        result = subprocess.run(docker_stats_cmd, capture_output=True, text=True, shell=True)
        print(result.stdout)
        return_list.append(json.loads(result.stdout))
    return return_list

def intermediate_data(container_name: str) -> pd.DataFrame:
    client = docker.from_env()
    containers = client.containers.list(filters={"ancestor": container_name})
    total_memory_usage = 0
    total_cpu_percentage = 0
    for container in containers:
        status = container.stats(decode=None, stream=False)
        try:
            total_memory_usage += status['memory_stats']['usage']
        except KeyError:
            total_memory_usage = 0
            total_cpu_percentage = 0
            break
        
        cpu_delta = status['cpu_stats']['cpu_usage']['total_usage'] - status['precpu_stats']['cpu_usage']['total_usage']
        system_delta = status['cpu_stats']['system_cpu_usage'] - status['precpu_stats']['system_cpu_usage']
        if system_delta > 0 and cpu_delta > 0:
            total_cpu_percentage += (cpu_delta / system_delta) * len(status['cpu_stats']['cpu_usage']['percpu_usage']) * 100

    total_memory_usage_mib = total_memory_usage / (1024 * 1024)  # Convert to MiB

    df = pd.DataFrame([[total_memory_usage_mib, total_cpu_percentage]], columns=['Total Memory Usage (MiB)', 'Total CPU Usage (%)'])
    print(df.to_string())
    return df


def dataframes_returner(numstream: str,
                        container: str) -> pd.DataFrame:
    """
    returns 1 df
    It also prints the information to the console.
    this loops until docker is down

    numstream: str
        The number of streams
    """
    df = pd.DataFrame(columns=['Time', 
                               'Total Memory Usage (MiB)', 
                               'Total CPU Usage (%)',
                               'Container Name'
                               ])  # Initialize DataFrame with columns
    start_time = datetime.now()  # Get the start time

    while True:
        if not docker_checker(numstream):
            break

        # Call the nvidia_scraper function
        info_list = nvidia_scraper()

        # right now its just gonna do the first gpu
        gpu_usage = float(info_list[0]['memory_usage'].split("MiB")[0])

        # docker_stats = docker_stats_grabber()


        cumulative = intermediate_data(container)

        # Get the current time and calculate elapsed seconds
        timestamp = datetime.now()
        elapsed_seconds = (timestamp - start_time).total_seconds()

        # Append the cumulative data to the DataFrame
        new_row = pd.DataFrame({'Time': [elapsed_seconds], 
                                'Total Memory Usage (MiB)': [cumulative['Total Memory Usage (MiB)'].values[0]], 
                                'Total CPU Usage (%)': [cumulative['Total CPU Usage (%)'].values[0]],
                                'GPU Memory Usage (MiB)': [gpu_usage],
                                'Container Name': container})
        df = pd.concat([df, new_row], ignore_index=True)

        # pprint(docker_stats)

    return df


def gpu_plotter(dataframe: pd.DataFrame,
                using_res: bool=True):
    """
    This function takes a DataFrame and plots 'Total Memory Usage (MiB)' and 'Total CPU Usage (%)' over 'Time' for each number of streams.
    It saves the plots to image files.

    dataframe: pd.DataFrame
        The DataFrame containing the data to plot
    using_res: bool
        if you arent benchmarking based off of resolution,
        make it False.
    """
    if not os.path.isdir('./figures'):
        Shell.mkdir('./figures')

    # Check the unique values of the dataframe's resolution value
    if using_res:
        unique_resolutions = dataframe['resolution'].unique()
        if len(unique_resolutions) != 1:
            raise ValueError("The dataframe contains more than one unique resolution value.")
        resolution = unique_resolutions[0]


    # Plot 'Total Memory Usage (MiB)' over 'Time' with separate lines for each number of streams
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Time', y='Total Memory Usage (MiB)', hue='total-streams', data=dataframe[dataframe['Container Name'] == container_name], palette='tab10')
    if using_res:
        plt.title(f'Total Memory Usage (RAM) Over Time for RTSP Streaming\nResolution: {resolution}')
    else:
        plt.title(f'Total Memory Usage (RAM) Over Time for RTSP Streaming')
    plt.grid(True)  # Add a grid
    plt.ylim(0, 40000)
    plt.xlabel("Time in seconds")
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1])
    if using_res:
        plt.savefig(f'./figures/ram_usage_over_time_{resolution}.png', bbox_inches='tight')
    else:
        plt.savefig(f'./figures/ram_usage_over_time.png', bbox_inches='tight')
    plt.clf()  # Clear the current figure

    # Plot 'Total CPU Usage (%)' over 'Time' with separate lines for each number of streams
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Time', y='Total CPU Usage (%)', hue='total-streams', data=dataframe[dataframe['Container Name'] == container_name], palette='tab10')
    if using_res:
        plt.title(f'Total CPU Usage Over Time for RTSP Streaming\nN cores times 100 (Resolution: {resolution})')
    else:
        plt.title(f'Total CPU Usage Over Time for RTSP Streaming\nN cores times 100')
    plt.grid(True)  # Add a grid
    plt.ylim(0, 3600)
    plt.xlabel("Time in seconds")
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1])
    if using_res:
        plt.savefig(f'./figures/cpu_usage_over_time_{resolution}.png', bbox_inches='tight')
    else:
        plt.savefig(f'./figures/cpu_usage_over_time.png', bbox_inches='tight')
    plt.clf()

    # gpu ram
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Time', y='GPU Memory Usage (MiB)', hue='total-streams', data=dataframe[dataframe['Container Name'] == container_name], palette='tab10')
    if using_res:
        plt.title(f'GPU RAM Usage (MiB) Over Time for RTSP Streaming\nResolution: {resolution}')
    else:
        plt.title(f'GPU RAM Usage (MiB) Over Time for RTSP Streaming')
    plt.grid(True)  # Add a grid
    plt.ylim(0, 14000)
    plt.xlabel("Time in seconds")
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1])
    if using_res:
        plt.savefig(f'./figures/gpu_ram_over_time_{resolution}.png', bbox_inches='tight')
    else:
        plt.savefig(f'./figures/gpu_ram_over_time.png', bbox_inches='tight')


def bar_plotter(tracks_with_res: dict,
                name: str = 'stream-tracks_by_resolution.png'):
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

def fps_finder(num_of_streams: int) -> dict:
    containers = Shell.run(f"docker ps -aq -f ancestor=fastmot-image | head -n {num_of_streams}").splitlines()
    fps_dict = {}
    for container in containers:
        output = Shell.run(f"docker logs {container}").splitlines()
        correct = Shell.find_lines_with(output, "Average FPS")[0]
        number = int(correct.split("FPS:")[-1].strip())
        fps_dict[container] = number
    return fps_dict



def main():

    #
    #
    #
    rtsp = True


    if rtsp:
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
    else:
        #phony
        vid1_list = [
            "/mnt/hdd/gvideo/25_2023-08-30_07-45-02.000-med-conflict.mp4",
        ]
        vid2_list = [
            "/mnt/hdd/gvideo/26_2023-08-30_07-45-02.000-med-conflict.mp4",
        ]

    mega_df = pd.DataFrame()
    tracks_with_res = {}
    fps_totals = {}

    for i, (vid, vid2) in enumerate(zip(vid1_list, vid2_list)):
        mega_df = pd.DataFrame()  # Initialize an empty DataFrame for each pair of videos
        print('Starting the fastmot launcher for videos', i)
        
        for num_of_streams in [1, 2, 3, 5, 7, 10, 11, 12]:
            print('Processing', num_of_streams, 'streams')
            time.sleep(1)
            
            res = fastmot_launcher(vid, vid2, num_of_streams, rtsp)

            df = dataframes_returner(num_of_streams)
            print("Past dataframes returner")
            # to all rows, add a column that says resolution, 
            # with value res
            
            df['resolution'] = res
            df['total-streams'] = num_of_streams
            mega_df = pd.concat([mega_df, df])
            print('HERERERERERERE')
            print(mega_df.to_string())

            fps_dict = fps_finder(num_of_streams)
            fps_totals[num_of_streams] = fps_dict

        print('Starting gpu plotter for videos', i)
        gpu_plotter(mega_df, True)

    pprint(tracks_with_res)
    print('Starting bar plotter')
    bar_plotter(tracks_with_res)

    print('-'*20)
    print('-'*20)
    print('-'*20)
    pprint(fps_totals)

    # Calculate the average FPS for each number of streams
    avg_fps = {num_streams: sum(fps_dict.values()) / len(fps_dict) for num_streams, fps_dict in fps_totals.items()}

    # Create lists of the number of streams and the corresponding average FPS
    num_streams = list(avg_fps.keys())
    fps_values = list(avg_fps.values())

    # Create a bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(num_streams, fps_values)
    plt.xlabel('Number of Streams')
    plt.ylabel('Average FPS')
    plt.title('Average FPS for Different Numbers of Streams')
    plt.savefig("average-fps-streams.png", bbox_inches='tight')


    print('Done')