import subprocess
import zipfile
import os

host = os.getenv("CVAT_host",None)
user = os.getenv("CVAT_user",None)
passwd = os.getenv("CVAT_passwd",None)
port = os.getenv("CVAT_port",None)

# placeholder tasks for now before deciding which tasks are useful.
tasks = [5, 6, 7, 8, 10, 11, 12, 13, 14, 66, 67, 68, 69, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 56, 57, 58, 60, 61, 64, 34, 37,38,39,40,41,42,43,44,46]
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
