import subprocess
import zipfile
import os

host = os.getenv("CVAT_host",None)
user = os.getenv("CVAT_user",None)
passwd = os.getenv("CVAT_passwd",None)
port = os.getenv("CVAT_port",None)

# placeholder tasks for now before deciding which tasks are useful.
tasks = [43,44]
os.makedirs("zips", exist_ok=True)
os.makedirs("unzips", exist_ok=True)

for task_number in tasks:

    output_zip = f"zips/{task_number}.zip"
    
    if os.path.exists(output_zip):
        continue

    subprocess.run([
        "cvat-cli",
        "--server-host", host,
        "--auth", f"{user}:{passwd}",
        "--server-port", port,
        "dump",
        "--format", "YOLO 1.1",
        "--with-images", "True",
        str(task_number),
        output_zip
    ])

    if os.path.exists(output_zip):
        unzip_folder = f"unzips/{task_number}_unzipped"
        if os.path.exists(unzip_folder):
            continue
        os.makedirs(unzip_folder, exist_ok=True)
        
        with zipfile.ZipFile(output_zip, 'r') as zip_ref:
            zip_ref.extractall(unzip_folder)
        
        print(f"Task {task_number} unzipped into {unzip_folder}")
    else:
        print(f"Error: {output_zip} not found.")
