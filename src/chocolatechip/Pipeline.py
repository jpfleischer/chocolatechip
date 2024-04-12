import os
from cloudmesh.common.Shell import Shell
from cloudmesh.common.console import Console
from pprint import pprint
import subprocess
import docker


def stop_everything():
    names = ["rabbitmq", 
             "nifi", 
             "fastmot", 
             "vid_dual", 
             "vid_online_clustering", 
             "vid_se", 
             "vid_statistics", 
             "vid_ttc",
             "rtsp_stream",
             "tracks_processing"]
    client = docker.from_env()

    for container in client.containers.list(all=True):
        if any(name in container.name for name in names):
            print(container.name)
            container.remove(force=True)



def rabbit(bundle_names: list,
           pipeline_dir: str = "/mnt/hdd/pipeline"):
    # rabbit setup

    allofthem = []
    
    for bundle in bundle_names:
        allofthem.append(("rabbitmq", f"rabbitmq-{bundle}", bundle))
    
    addition = 0
    for software, full, intersection in allofthem:
        Console.info(f"Building {software}")        
        result = subprocess.run(f"cd {os.path.join(pipeline_dir, software)} && make CUSTOM_NAME={full} "\
                                f"NETWORK_NAME={intersection} "\
                                f"MAIN_PORT={str(15672 + addition)} "\
                                f"PORT2={str(5672 + addition)} "                                
                                ,
            shell=True)
        print(result)
        addition += 1


def tracks_processing(bundle_names: list,
           pipeline_dir: str = "/mnt/hdd/pipeline"):
    # tracks processing setup

    allofthem = []
    
    for bundle in bundle_names:
        allofthem.append(("tracks_processing", f"tracks_processing-{bundle}-1", bundle))
        allofthem.append(("tracks_processing", f"tracks_processing-{bundle}-2", bundle))
    
    flip = True
    for software, full, intersection in allofthem:
        Console.info(f"Building {software}")        
        if flip:
            queue = "dt_tracks_ready_1"
        else:
            queue = "dt_tracks_ready_2"
        result = subprocess.run(f"cd {os.path.join(pipeline_dir, software)} && make CUSTOM_NAME={full} "\
                                f"NETWORK_NAME={intersection} "\
                                f"RBBT_IP=rabbitmq-{intersection} "\
                                f"QUEUE={queue}"
                                ,
            shell=True)
        print(result)
        flip = not flip


def rtsp(bundle_names: list,
         pipeline_dir: str = "/mnt/hdd/pipeline"):
    # rtsp setup

    allofthem = []
    
    for bundle in bundle_names:
        allofthem.append(("rtsp_stream", f"rtsp_stream-{bundle}-1", bundle))
        allofthem.append(("rtsp_stream", f"rtsp_stream-{bundle}-2", bundle))
    
    addition = 0
    for software, full, intersection in allofthem:
        Console.info(f"Building {software}")        
        result = subprocess.run(f"cd {os.path.join(pipeline_dir, software)} && make CUSTOM_NAME={full} "\
                                f"NETWORK_NAME={intersection} "\
                                f"PORT_1={str(8554 + addition)} "\
                                f"PORT_2={str(1935 + addition)} "
                                ,
            shell=True)
        print(result)
        addition += 1


def nifi(bundle_names: list,
           pipeline_dir: str = "/mnt/hdd/pipeline"):
    # nifi setup

    allofthem = []
    
    for bundle in bundle_names:
        allofthem.append(("nifi", f"nifi-{bundle}", bundle))
    
    addition = 0
    for software, full, intersection in allofthem:
        Console.info(f"Building {software}")        
        result = subprocess.run(f"cd {os.path.join(pipeline_dir, software)} && make CUSTOM_NAME={full} "\
                                f"NETWORK_NAME={intersection} "\
                                f"MAIN_PORT={str(30105 + addition)} "\
                                f"PORT2={str(6342 + addition)} "\
                                f"RBBT_IP=rabbitmq-{intersection}"
                                ,
            shell=True)
        print(result)
        result = subprocess.run(f"cd {os.path.join(pipeline_dir, software)} && make patch CUSTOM_NAME={full} "\
                                f"NETWORK_NAME={intersection} "\
                                f"MAIN_PORT={str(30105 + addition)} "\
                                f"PORT2={str(6342 + addition)} "\
                                f"RBBT_IP=rabbitmq-{intersection}"          
                                ,
                                shell=True)
        print(result)
        addition += 1

    

def network_checker(name_of_network: str) -> bool:
    try:
        r = Shell.run(f"docker network ls | grep {name_of_network}")
        return True    
    except RuntimeError:
        pass

    try:
        Shell.run(f"docker network create {name_of_network}")
        return True
    except RuntimeError:
        return False
    

def main():
    """
    this restarts the pipeline
    """
    Console.info("Stopping containers...")
    stop_everything()

    bundle_names = [
        "3334",
        "3032",
        "3248",
        "3265",
    ]

    pipeline_dir = "/mnt/hdd/pipeline"


    Console.info("Building networks...")
    # build network
    for bundle in bundle_names:
        network_checker(bundle)


    # rabbit setup
    rabbit(bundle_names, pipeline_dir=pipeline_dir)
    # nifi setup
    nifi(bundle_names, pipeline_dir=pipeline_dir)

    # tracks processing setup
    tracks_processing(bundle_names, pipeline_dir=pipeline_dir)

    # rtsp setup
    rtsp(bundle_names, pipeline_dir=pipeline_dir)

    softwares = [
                 "vid_dual",
                 "vid_online_clustering",
                 "vid_se",
                 "vid_statistics",
                 "vid_ttc"
                 ]
    lookup_name = []


    for bundle in bundle_names:
        for software in softwares:
            lookup_name.append((software, f"{software}-{bundle}", bundle))

    pprint(lookup_name)
    for software, full, intersection in lookup_name:
        Console.info(f"Building {software}")        
        result = subprocess.run(
            f"cd {os.path.join(pipeline_dir, software)} && "\
            f"make CUSTOM_NAME={full} "\
            f"NETWORK_NAME={intersection} "\
            f"RBBT_IP=rabbitmq-{intersection}",
            shell=True
        )
        
        print(result)
        if result.returncode != 0:
            Console.error("Failure")
            break
    

