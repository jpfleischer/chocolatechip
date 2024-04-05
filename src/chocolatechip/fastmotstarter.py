import subprocess
import os
import docker

def main():
    client = docker.from_env()
    names = ['fastmot']

    for container in client.containers.list(all=True):
        if any(name in container.name for name in names):
            print(container.name)
            container.remove(force=True)

    pipeline_dir = "/mnt/hdd/pipeline"
    name_fastmot = 'fastmot-image'
    intersection = '3334'
    result = subprocess.run(f"cd {os.path.join(pipeline_dir, "fastmot")} && make run "\
                            f"CUSTOM_NAME={name_fastmot} "\
                            f"NETWORK_NAME={intersection} "\
                            f"RBBT_IP=rabbitmq-{intersection}"
                            ,
        shell=True)
    print(result)