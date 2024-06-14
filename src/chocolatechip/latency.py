import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from chocolatechip import fastmotstarter
import docker
from datetime import datetime
from pprint import pprint

def latency_benchmark(pairs: int):
    # hardcoded only 1
    intersection = ['3334', '3032', '3248', '3265']
    intersection = intersection[:pairs]

    # make sure only set1 is in the fastmotstarter
    fastmotstarter.main(pairs)
    client = docker.from_env()

    list_of_containers = ['fastmot', 
                        'tracks_processing', 
                        'vid_online_clustering',
                        'vid_ttc',
                        'vid_statistics',
                        'vid_se']
 

    # print(full_paths)
    ids = []
                
     # List of container names to search for
    list_of_images = ['fastmot-image-1', 'fastmot-image-2',
                      'fastmot-image-3', 'fastmot-image-4',
                      'fastmot-image-5', 'fastmot-image-6',
                      'fastmot-image-7', 'fastmot-image-8',]
    
    list_of_images = list_of_images[:pairs*2]
    

    # Get all containers
    all_containers = client.containers.list(all=True)
    for container in all_containers:
        for name in list_of_containers:
            if name in container.name and any(x in container.name for x in intersection):
                ids.append({'name': container.name,
                            'id': container.id})
    for image_name in list_of_images:
        # Filter the list to get the containers with the specified name
        image_containers = [container for container in all_containers if image_name in container.name]

        # Check if there are any containers with the specified name
        if image_containers:
            # Get the last container that ran with the specified name
            last_image_container = image_containers[0]

            # Append the container to the 'ids' list
            ids.append({'name': last_image_container.name, 'id': last_image_container.id})

    pprint(ids)

    logs = []
    # use docker api to print logs
    for iterated_id in ids:

        logs.append(client.containers.get(iterated_id['id']).logs().decode('utf-8'))

    singleton = "\n".join(logs)
    print(singleton[-1000:-1])
    timers = []
    for line in singleton.splitlines():
        if "[TIME]" in line:
            # just a little patch because i dont want
            # that right now
            if 'Waiting for messages' not in line:
                timers.append(line)

    # pprint(timers)
    # scratch doesnt matter
    # pprint(timers)
    time_dict = {}
    temp_id = 0

    # Loop through the list of timestamps
    for time in timers:
        time_split = time.split('[TIME]')
        name = time_split[0].strip()
        time_val = time_split[-1].strip()
        # if 'TTC' in name:
            # print(name)
        try:
            dt = datetime.strptime(time_val, '%Y-%m-%d %H:%M:%S.%f')
            # dt = dt.replace(tzinfo=None)  # remove timezone info
        except ValueError as e:
            print("error", name)
            print(e)
            # print(name)
        time_dict[temp_id] = {'name': name, 'time': dt}
        temp_id += 1

    # Sort the dictionary based on the 'time' key
    sorted_time_dict = {idx: time_dict[idx] for idx in sorted(time_dict, key=lambda x: time_dict[x]['time'])}

    new_timers = []
    for key in sorted(sorted_time_dict.keys(), key=lambda k: sorted_time_dict[k]['time']):
        new_timers.append(sorted_time_dict[key])

    # new_timers is a list of dicts with a value of
    # name and i want to change the name to correct it, for example
    # {'2024-01-05 13:20:00 [    INFO] Starting video capture',
    # is wrong i just want it to be Starting video capture
    for timer in new_timers:
        if ('Starting video capture' in timer['name']) or ('Finishing video capture' in timer['name']):
            timer['name'] = timer['name'].split('INFO]')[-1].strip()
            print('.', end='')

    # pprint(new_timers)
    subset_list = []
    found_start = False

    for entry in new_timers:
        if found_start:
            subset_list.append(entry)
        elif 'Starting video capture' in entry['name']:
            subset_list.append(entry)
            found_start = True

    # Printing the subset list
    tuple_list = [(timer['name'], timer['time'].strftime('%Y-%m-%d %H:%M:%S.%f')) for timer in subset_list]

    # Sort the list based on the datetime
    sorted_tuple_list = sorted(tuple_list, key=lambda x: x[1])
    pprint(sorted_tuple_list)
    pprint(subset_list)
    # take the list of dicts and take the name
    # of each one and print the total unique names
    names = []
    for entry in subset_list:
        names.append(entry['name'])
    pprint(set(names))

    # Creating lists for start times, end times, and durations
    start_times = [event['time'] for event in subset_list]
    end_times = start_times[1:] + [start_times[-1]]  # Shift end times by 1 to align with start times
    durations = [(end - start).total_seconds() for start, end in zip(start_times, end_times)]


    cleaned_names = {
        'Starting video capture': 'Started video capture (F)',
        'Finishing video capture': 'Finished video capture (F)',
        'I am now beginning to process the video, the time is': 'Beginning video processing (TP)',
        'Loaded from csv at': 'CSV data load (TP)',
        'I just sent it to the Rabbit queue at': 'Sent to rabbit queue (TP)',
        'Started processing data at': 'Processing started (VOC)',
        'Begin writing to db at': 'Started writing to DB (TP)',
        'Finish writing to db (final step) at': 'Finished writing to DB (TP)',
        'Finished processing data at': 'Processing finished (VOC)',
        'Started TTC computation at': "Started TTC Collision Analysis (VT)",
        'Finished TTC computation at': "Finished TTC Collision Analysis (VT)",
        'Started SE computation at': "Started SE Analysis (SE)",
        'Finished SE computation at': "Finished SE Analysis (SE)",
        'Started stats analysis at': "Started Statistics Analysis (VS)",
        'Finished stats analysis at': "Finished Statistics Analysis (VS)",
        # 'Vid clustering online': 'Vid clustering online',
        # ... (add more cleaned names as needed)
    }

    cleaned_data = []

    for item in subset_list:
        name = item['name']
        cleaned_name = cleaned_names.get(name, name)
        cleaned_data.append({'name': cleaned_name, 'time': item['time']})

    # cleaned data is a list of dicts. i want to pprint the
        # unique name values in each dictionary
    pprint({entry['name'] for entry in cleaned_data})

    # Create a color map for different processes
    colors = ['blue', 
            'green', 
            'red', 
            'orange', 
            'purple', 
            'yellow', 
            'cyan', 
            'magenta',
            'brown',
            'pink',
            'gray',
            'olive',
            'lime',
            'teal',
            'coral']
    color_map = {process: color for process, color in zip(cleaned_names.values(), colors)}

    # Create a shape map for different processes
    shapes = ['o', 'o', 'p', '<', '>', 'x', '^', 'v', 'X', 'p', 'h', 'd', 'D', 'P', '+']
    shape_map = {process: shape for process, shape in zip(cleaned_names.values(), shapes)}


    # Group timestamps by process
    processes = {}
    for item in cleaned_data:
        log = item['name']
        time = item['time']
        
        # Extracting the process name without timestamp and log level
        try:
            name = log.split('] ')[1]
        except IndexError:
            name = log  # If splitting fails, use the entire log entry as the name
        
        if name in processes:
            processes[name].append(time)
        else:
            processes[name] = [time]
    pprint(processes)

    new_processes = {process: [str(timestamp) for timestamp in timestamps] for process, timestamps in processes.items()}
    # pprint(new_processes)
    # Create a list of tuples from the dictionary
    tuple_list = [(process, timestamp) for process, timestamps in new_processes.items() for timestamp in timestamps]

    # Sort the list based on the timestamps
    sorted_tuple_list = sorted(tuple_list, key=lambda x: x[1])
    # print('\n\n')
    # pprint(sorted_tuple_list)


    # Convert the list of tuples to a DataFrame
    df = pd.DataFrame(sorted_tuple_list, columns=['Process', 'Timestamp'])

    # Convert the 'Timestamp' column to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Sort the DataFrame based on the 'Timestamp' column
    df.sort_values('Timestamp', inplace=True)

    # Reset the index of the DataFrame
    df.reset_index(drop=True, inplace=True)

    # print(df)
    # Add a new column 'pairs' to the DataFrame and set it to the value of pairs
    df = df.assign(pairs=pairs)

    # Initialize the base filename and the counter
    base_filename = 'latency'
    counter = 0

    # Construct the initial filename
    filename = f'{base_filename}.csv'

    # While the file exists, increment the counter and construct a new filename
    while os.path.isfile(filename):
        counter += 1
        filename = f'{base_filename}{counter}.csv'

    # Write the DataFrame to the file
    df.to_csv(filename, index=False)
    # Create a color map for different processes using a seaborn palette
    color_map = {process: sns.color_palette("husl", len(df['Process'].unique()))[i] for i, process in enumerate(df['Process'].unique())}

    shapes = ['o', 'o', 'p', '<', '>', 'x', '^', 'v', 'X', 'p', 'h', 'd', 'D', 'P', '+']
    shape_map = {process: shape for process, shape in zip(cleaned_names.values(), shapes)}

    # Plotting code using the updated color map and marker map
    plt.figure(figsize=(8, 5))

    # Create a list to store the order of the processes
    process_order = []

    for i, process in enumerate(df['Process'].unique()):
        timestamps = df[df['Process'] == process]['Timestamp'].tolist()
        timestamps.sort()  # Sort timestamps in ascending order
        y_values = [i + 1] * len(timestamps)  # Assign a y-value to each timestamp for plotting
        plt.scatter(timestamps, y_values, label=process, 
                    color=color_map[process], marker=shape_map[process], s=150)
        process_order.append(process)  # Add the process to the list

    # Beautify the plot
    plt.title('Processes Over Time for Dual 120s RTSP Stream')
    plt.xlabel('Time')
    plt.ylabel('Processes')
    plt.legend(loc='upper left', 
               bbox_to_anchor=(1.05, 0.25), 
               labelspacing=-2.5, frameon=False)
    plt.yticks(range(1, len(df['Process'].unique()) + 1), process_order)  # Use the process_order list for the y-axis labels

    # Determine the range of your data
    x_min = min(min(timestamps) for timestamps in processes.values())
    x_max = max(max(timestamps) for timestamps in processes.values())

    # Set the frequency of the x-ticks (e.g., every 30 minutes)
    # x_tick_freq = '45s'
    # x_tick_freq = '5s'
    x_tick_freq = '8s'

    # Extend x_max to the next tick
    x_max_extended = x_max + pd.Timedelta(x_tick_freq)

    # Generate the x-ticks
    x_ticks = pd.date_range(start=x_min, end=x_max_extended, freq=x_tick_freq)

    # Set the x-ticks
    plt.xticks(x_ticks, x_ticks.strftime('%H:%M:%S'), rotation=45, ha='right')

    plt.grid(axis='y')
    plt.tight_layout()
    # plt.subplots_adjust(right=0.8)

    # Add text to the plot
    plt.text(1.15, 0.1, 'F - fastmot\n'
                        'TP - tracks_processing\n'
                        'VOC - vid_online_clustering\n'
                        'VT - vid_ttc\n'
                        'VS - vid_statistics\n'
                        'SE - vid_se\n',
            horizontalalignment='left', verticalalignment='top', 
            transform=plt.gca().transAxes)
    plt.savefig('./latency.png', bbox_inches='tight')


def main():
    fastmotstarter.fastmot_cleaner()

    for number_of_paired_streams in [1, 2, 3, 4]:
        # 1 2 3 4
        latency_benchmark(number_of_paired_streams)
        fastmotstarter.fastmot_cleaner()