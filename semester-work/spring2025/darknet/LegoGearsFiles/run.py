import csv
from datetime import datetime
from cloudmesh.common.StopWatch import StopWatch
from cloudmesh.gpu.gpu import Gpu
import subprocess
import train_setup  # links training data to its labels
import os
import getpass
import glob
import shutil
import re
import json
from threading import Thread, Event
import unicodedata

from pathlib import Path
import zipfile
from hw_info import summarize_env


uva_running = os.environ.get("UVA_VIRGINIA_RUNNING", "false").lower() == "true"

def human_readable_size(num_bytes):
    """
    Convert a size in bytes to a human-readable string.
    """
    try:
        num_bytes = float(num_bytes)
    except (ValueError, TypeError):
        return "N/A"
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']:
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} PiB"


def is_wsl():
    """
    Detect if running in Windows Subsystem for Linux (WSL)
    """
    try:
        with open("/proc/version", "r") as f:
            version_info = f.read()
        # In WSL, /proc/version usually contains "Microsoft" or "WSL"
        if "Microsoft" in version_info or "WSL" in version_info:
            return True
    except Exception:
        pass
    return False


def get_disk_info():
    """
    Returns disk info for the drive containing the current terminal's working directory.
    
    Keys:
      - Disk Capacity
      - Disk Model

    If the environment variable WINDOWS_HARD_DRIVE is set (e.g., when using Git Bash on Windows),
    its value is used as the Disk Model and the environment variable WINDOWS_HARD_DRIVE_CAPACITY (if set)
    is converted from bytes into a human-readable format and used as the Disk Capacity.
    Otherwise, the function attempts to determine disk info using Linux commands.
    """
    try:
        # If running on Windows with the environment variable already set, use it.
        windows_hd = os.environ.get("WINDOWS_HARD_DRIVE")
        if windows_hd:
            windows_cap = os.environ.get("WINDOWS_HARD_DRIVE_CAPACITY", "N/A")
            # Convert capacity from bytes (as a string) to human-readable format.
            windows_cap_hr = human_readable_size(windows_cap) if windows_cap != "N/A" else "N/A"
            return {
                "Disk Capacity": windows_cap_hr,
                "Disk Model": windows_hd
            }
        
        # For non-Windows systems (or if the environment variable is not set), use Linux commands.
        cwd = os.getcwd()
        # Get the device for the current working directory using 'df'
        df_cmd = f"df --output=source {cwd}"
        df_output = subprocess.check_output(df_cmd, shell=True, text=True).strip().splitlines()
        if len(df_output) < 2:
            raise ValueError("Could not determine device from df output")
        current_device = df_output[1].strip()

        # List physical disks (not partitions) with fields: NAME, TYPE, SIZE, MODEL.
        lsblk_cmd = "lsblk -d -o NAME,TYPE,SIZE,MODEL -n"
        lsblk_output = subprocess.check_output(lsblk_cmd, shell=True, text=True).strip()
        disk_info_list = [[], []]  # [non-matches, matches]

        for line in lsblk_output.splitlines():
            parts = line.split()
            if len(parts) < 4:
                continue
            name = parts[0]
            # Check if this disk name is found in the current device path.
            matches = name in current_device
            dev_type = parts[1]
            size = parts[2]
            model = " ".join(parts[3:])
            if dev_type != "disk":
                continue
            # Append to the matched list if the device name is found in current_device,
            # otherwise to the non-matches list.
            disk_info_list[1 if matches else 0].append({
                "size": size,
                "model": model
            })

        # Prefer the disks that matched the current device.
        if disk_info_list[1]:
            disk_capacities = ", ".join(disk["size"] for disk in disk_info_list[1])
            disk_models = ", ".join(disk["model"] for disk in disk_info_list[1])
        elif disk_info_list[0]:
            disk_capacities = ", ".join(disk["size"] for disk in disk_info_list[0])
            disk_models = ", ".join(disk["model"] for disk in disk_info_list[0])
        else:
            disk_capacities = disk_models = "N/A"

    except Exception as e:
        print("Error retrieving disk info:", e)
        disk_capacities = disk_models = "N/A"

    return {
        "Disk Capacity": disk_capacities,
        "Disk Model": disk_models,
    }


def run_fio_speed_test(test_file="fio_test_file", block_size="1M", runtime=20, size="1G"):
    """
    Uses fio to measure disk sequential write and read speeds.
    
    The test uses direct I/O and runs for the specified runtime (in seconds)
    with a given file size, returning a tuple: (write_speed, read_speed)
    formatted as strings in MiB/s.
    
    Note:
      - The write test creates/overwrites the test file.
      - The read test uses the file created by the write test.
      - The test file is removed after the tests.
    """
    # Sequential Write Speed Test
    try:
        write_cmd = (
            f"fio --name=seqwrite --ioengine=libaio --direct=1 --rw=write "
            f"--bs={block_size} --runtime={runtime} --time_based --size={size} "
            f"--filename={test_file} --output-format=json"
        )
        write_result = subprocess.run(write_cmd, shell=True, capture_output=True, text=True)
        write_output = json.loads(write_result.stdout)
        # Extract write bandwidth (in KiB/s) and convert to MiB/s
        write_bw_kib = write_output["jobs"][0]["write"]["bw"]
        write_bw_mib = write_bw_kib / 1024
        write_speed = f"{write_bw_mib:.2f} MiB/s"
    except Exception as e:
        print("Error during write test:", e)
        write_speed = "N/A"

    # Sequential Read Speed Test
    try:
        read_cmd = (
            f"fio --name=seqread --ioengine=libaio --direct=1 --rw=read "
            f"--bs={block_size} --runtime={runtime} --time_based --size={size} "
            f"--filename={test_file} --output-format=json"
        )
        read_result = subprocess.run(read_cmd, shell=True, capture_output=True, text=True)
        read_output = json.loads(read_result.stdout)
        # Extract read bandwidth (in KiB/s) and convert to MiB/s
        read_bw_kib = read_output["jobs"][0]["read"]["bw"]
        read_bw_mib = read_bw_kib / 1024
        read_speed = f"{read_bw_mib:.2f} MiB/s"
    except Exception as e:
        print("Error during read test:", e)
        read_speed = "N/A"

    # Clean up the test file
    try:
        os.remove(test_file)
    except Exception:
        pass

    print(write_speed, read_speed)
    return write_speed, read_speed


def slugify(text: str, allowed: str = "-_.") -> str:
    """
    Make a filesystem-friendly string:
    - ASCII only (strip accents)
    - Replace spaces with underscores
    - Any char not alnum or in `allowed` -> underscore
    - Collapse multiple underscores and trim punctuation
    """
    s = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
    s = s.replace(" ", "_")
    s = re.sub(fr"[^A-Za-z0-9{re.escape(allowed)}]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("._-")
    return s[:180]  # keep it reasonable


if __name__ == "__main__":
    username = getpass.getuser()
    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get GPU name
    # Step 2: Get GPU info
    gpu = Gpu()
    all_vram = [card['fb_memory_usage']['total'] for card in gpu._smi]
    all_gpu_names = [card["product_name"] for card in gpu.system()]

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cuda_visible:
        print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
        try:
            indices = [int(x.strip()) for x in cuda_visible.split(",") if x.strip() != ""]
        except ValueError as e:
            print(f"Error parsing CUDA_VISIBLE_DEVICES: {e}")
            indices = []
    else:
        print("I CANT SEE CUDA. Attempting to query nvidia-smi...")
        try:
            # Run the command and capture the output
            cmd = "nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -"
            indices_str = subprocess.check_output(cmd, shell=True, text=True).strip()
            print(f"nvidia-smi returned: {indices_str}")
            indices = [int(x.strip()) for x in indices_str.split(",") if x.strip() != ""]
        except Exception as e:
            print(f"Failed to get GPU indices from nvidia-smi: {e}")
            indices = []

    if indices:
        selected_vram = [all_vram[i] for i in indices if i < len(all_vram)]
        selected_gpu_names = [all_gpu_names[i] for i in indices if i < len(all_gpu_names)]
        vram = ", ".join(selected_vram)
        gpu_name = ", ".join(selected_gpu_names)
        if uva_running:
            gpus_str = ",".join(str(i) for i in indices)
        else:
            gpus_str = ",".join(str(i) for i in range(len(indices)))
    else:
        # Fallback: use all available info if no indices found
        vram = " ".join(all_vram)
        gpu_name = " ".join(all_gpu_names)
        gpus_str = ""

    # Create safe strings for filenames
    gpu_name_safe = slugify(gpu_name.replace(",", "-"))  # turn commas into a separator, then slugify
    # We'll set cpu_name_safe later after we obtain sysinfo

    # Step 3: If running in Apptainer, create an output folder and change into it.
    # This ensures that files generated by the darknet subprocess go into this directory.
    if "APPTAINER_ENVIRONMENT" in os.environ:
        print("Running in Apptainer environment")
    elif os.path.exists('/.dockerenv'):
        print("Running in Docker")
        username = os.environ.get("TRUE_USER", "default_username")
    else:
        print("Running non-apptainer")
        darknetloc = 'darknet'

    output_dir = os.path.join("/outputs", f"benchmark__{username}__{gpu_name_safe}__{now}")
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)
    print(f"Changed directory to output folder: {output_dir}")
    darknetloc = '/host_workspace/darknet/build/src-cli/darknet'

    
    # --- start GPU watcher (background thread) ---
    gpu_watch_thread = None
    watch_log = os.path.join(output_dir, "mylogfile.log")
    watch_stop = Event()

    try:
        if gpu.count > 0:
            gpu_watch_thread = Thread(target=gpu.watch, kwargs={
                "logfile": watch_log,
                "delay": 1.0,
                "dense": True,
                # monitor the GPUs you've selected; None means "all visible"
                "gpu": indices if indices else None,
                "install_signal_handler": False,   # important in threads
                "stop_event": watch_stop,          # allows clean shutdown
            })
            gpu_watch_thread.daemon = True
            gpu_watch_thread.start()
            print(f"GPU watcher logging to {watch_log}")
        else:
            print("No GPUs detected; skipping GPU watch.")
    except Exception as e:
        print(f"Skipping GPU watch: {e}")
    # --- end GPU watcher start ---

    # Decide which GPU indices Darknet should see
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if indices:
        if cuda_visible:
            # indices are relative to the visible set
            gpus_str = ",".join(str(i) for i in range(len(indices)))
        else:
            # use absolute indices
            gpus_str = ",".join(str(i) for i in indices)
    else:
        gpus_str = ""  # fallback: let Darknet decide / single GPU

    try:
        # Step 5: Run benchmark / training
        StopWatch.start("benchmark")
        cmd = (
            f"{darknetloc} detector -map -dont_show -nocolor "
            + (f"-gpus {gpus_str} " if gpus_str else "")
            + "train /workspace/LegoGears_v2/LegoGears.data "
            "/workspace/LegoGears_v2/LegoGears.cfg "
            "2>&1 | tee training_output.log"
        )
        subprocess.call(cmd, shell=True)
        StopWatch.stop("benchmark")
    finally:
        # --- stop GPU watcher ---
        if gpu_watch_thread is not None:
            try:
                watch_stop.set()     # tell watch() to exit
                gpu.running = False  # belt & suspenders
                gpu_watch_thread.join(timeout=3)
                print(f"GPU watcher stopped; log at {watch_log}")
            except Exception as e:
                print(f"Error stopping GPU watcher: {e}")
        # --- end stop ---


    benchmark_result = StopWatch.get_benchmark()

    # Step 6: Extract sysinfo and benchmark results
    sysinfo = benchmark_result["sysinfo"]
    benchmark = benchmark_result["benchmark"]["benchmark"]

    # Now that we have sysinfo, create a safe CPU name string.
    cpu_name_safe = slugify(sysinfo["cpu"])

    # Get disk information using lsblk.
    print("Getting disk information")
    disk_info = get_disk_info()

    # Run dd speed tests to get file-based write and read speeds.
    print("Running disk speed test")
    dd_write_speed, dd_read_speed = run_fio_speed_test()

    # Gather CUDA/cuDNN/GPU info
    env = summarize_env(indices=indices, training_log_path=os.path.join(output_dir, "training_output.log"))

    data = {
        "Benchmark Time (s)": benchmark["time"],
        "CPU Name": sysinfo["cpu"],
        "CPU Threads": sysinfo["cpu_threads"],
        "GPU Name": gpu_name,
        "GPU VRAM": vram,
        "Total Memory": sysinfo["mem.total"],
        "OS": "WSL" if is_wsl() else sysinfo["uname.system"],
        "Architecture": sysinfo["uname.machine"],
        "Python Version": sysinfo["python.version"],
        "Disk Capacity": disk_info["Disk Capacity"],
        "Disk Model": disk_info["Disk Model"],
        "Write Speed": dd_write_speed,
        "Read Speed": dd_read_speed,
        "Working Dir": os.getenv("ACTUAL_PWD", "N/A"),
        "CUDA Version": env["cuda_version"],
        "cuDNN Version": env["cudnn_version"],
        "GPUs Used": env["num_gpus_used"],
        "Compute Capability": env["compute_caps_str"],
    }

    # Step 7: Create a unique CSV filename in the current (output) directory
    csv_name = f"benchmark__{username}__{gpu_name_safe}__{cpu_name_safe}__{now}.csv"
    csv_path = os.path.join(output_dir, csv_name)
    with open(csv_path, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())
        writer.writeheader()
        writer.writerow(data)
    print(f"Benchmark results saved to {csv_path}")

    
    # Move all files matching "*weights" from /workspace/LegoGears_v2/ to /outputs
    for file in glob.glob("/workspace/LegoGears_v2/*weights"):
        shutil.move(file, output_dir)


    bundle_name = f"benchmark_bundle__{username}__{gpu_name_safe}__{cpu_name_safe}__{now}.zip"
    bundle_path = Path(output_dir) / bundle_name

    with zipfile.ZipFile(bundle_path, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in Path(output_dir).rglob("*"):
            if not p.is_file():
                continue
            if p.name == bundle_name:
                continue  # don't include the zip itself
            if p.name.lower().endswith(".weights"):
                continue  # exclude Darknet weight files
            z.write(p, arcname=p.relative_to(output_dir))

    print(f"Zipped outputs to: {bundle_path}")
