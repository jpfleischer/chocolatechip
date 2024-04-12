import time
import subprocess
import os
import docker
from chocolatechip import Stream
import yaspin
import pandas as pd

def fastmot_cleaner():
    client = docker.from_env()
    names = ['fastmot']

    for container in client.containers.list(all=True):
        if any(name in container.name for name in names):
            print(container.name)
            container.remove(force=True)


def main():
    fastmot_cleaner()
    
    pipeline_dir = "/mnt/hdd/pipeline"
    name_fastmot = 'fastmot-image'

    list_of_dictionaries = []

    set_1 = {
        'intersection': '3334',
        'video_1': '/mnt/huge/BrowardVideos/31_2023-08-31_09-30-02.000-med-conflict.mp4',
        'video_2': '/mnt/huge/BrowardVideos/32_2023-08-31_09-30-02.000-med-conflict.mp4',

        'cam_id1': '31',
        'cam_id2': '32',
    }

    set_2 = {
        'intersection': '3032',
        'video_1': '/mnt/huge/BrowardVideos/21_2023-08-30_08-45-03.000-med-conflict.mp4',
        'video_2': '/mnt/hdd/gvideo/22_2023-08-30_08-45-03.000-med-conflict.mp4',

        'cam_id1': '21',
        'cam_id2': '22',
    }

    set_3 = {
        'intersection': '3248',
        'video_1': '/mnt/huge/BrowardVideos/25_2023-08-30_07-45-02.000-med-conflict.mp4',
        'video_2': '/mnt/huge/BrowardVideos/26_2023-08-30_07-45-02.000-med-conflict.mp4',
        'cam_id1': '25',
        'cam_id2': '26',
    }

    # something about the 34 35 resolution doesnt work yet
    # gotta make one for 28 29
    set_4 = {
        'intersection': '3265',
        'video_1': '/mnt/huge/BrowardVideos/28_2023-08-23_19-45-03.000-med-conflict.mp4',
        'video_2': '/mnt/huge/BrowardVideos/29_2023-08-23_19-45-03.000-med-conflict.mp4',

        'cam_id1': '28',
        'cam_id2': '29',
    }

    megaset = {
        2: [set_1],
        4: [set_1, set_2],
        6: [set_1, set_2, set_3],
        8: [set_1, set_2, set_3, set_4],
               }
    
    # list_of_dictionaries.append(set_1)
    # list_of_dictionaries.append(set_2)
    mega_df = pd.DataFrame()

    container_name = 'fastmot-image'

    for key, value in megaset.items():
        
        index_1 = 1
        index_2 = 2
        for dictionary in value:
            intersection = dictionary['intersection']
            video_1 = dictionary['video_1']
            video_2 = dictionary['video_2']
            cam_id1 = dictionary['cam_id1']
            cam_id2 = dictionary['cam_id2']
            result = subprocess.Popen(f"cd {os.path.join(pipeline_dir, "fastmot")} && make run "\
                                f"CUSTOM_NAME={name_fastmot}-{str(index_1)} "\
                                f"CUSTOM_NAME_2={name_fastmot}-{str(index_2)} "\
                                f"NETWORK_NAME={intersection} "\
                                f"RBBT_IP=rabbitmq-{intersection} "\
                                f"VID={video_1} "\
                                f"VID2={video_2} "\
                                f"CAM_ID={cam_id1} "\
                                f"CAM_ID2={cam_id2} "\
                                f"PORT_1={str(8554+(index_1-1))} "\
                                f"PORT_2={str(8554+(index_2-1))}"
                                ,
            shell=True)
            print(result)
            print('#'*50)
            print(str(8554+(index_1-1)), str(8554+(index_2-1)))
            index_1 += 2
            index_2 += 2
        while True:
            if Stream.docker_checker(key):
                print('its up')
                break
            else:
                time.sleep(1)
                print('.', end='', flush=True)
                continue
            
                
        # 2 just signifies number of fastmot containers
        # for each intersection.
        df = Stream.dataframes_returner(key,
                                        container_name)
        df['total-streams'] = key
        mega_df = pd.concat([mega_df, df])

        with yaspin.yaspin() as sp:
            sp.text = "Waiting for 10 seconds"
            time.sleep(10)
        fastmot_cleaner()

    mega_df.to_csv(f'./figures/{container_name}.csv')

    Stream.gpu_plotter(mega_df, False)