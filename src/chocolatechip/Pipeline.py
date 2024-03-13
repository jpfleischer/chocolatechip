import os
from cloudmesh.common.Shell import Shell
from cloudmesh.common.console import Console


def main():
    """
    this restarts the pipeline
    """
    pipeline_dir = "/mnt/hdd/pipeline"
    softwares = ["tracks_processing",
                 "vid_dual",
                 "vid_online_clustering",
                 "vid_se",
                 "vid_statistics",
                 "vid_ttc"]
    for software in softwares:
        Console.info(f"Building {software}")        
        result = Shell.run(f"cd {os.path.join(pipeline_dir, software)} && make")
        print(result)