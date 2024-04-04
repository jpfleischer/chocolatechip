import subprocess
import os

def main():
    pipeline_dir = "/mnt/hdd/pipeline"
    name_fastmot = 'fastmot-image'
    intersection = '3032'
    result = subprocess.run(f"cd {os.path.join(pipeline_dir, "fastmot")} && make run "\
                            f"CUSTOM_NAME={name_fastmot} "\
                            f"NETWORK_NAME={intersection} "\
                            f"RBBT_IP=rabbitmq-{intersection}"
                            ,
        shell=True)
    print(result)