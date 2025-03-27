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

def get_disk_info():
    """
    Returns disk info for the drive containing the current terminal's working directory.
    Keys:
      - Disk Names
      - Disk Capacities
      - Disk Models
    """
    try:
        # Get current working directory.
        cwd = os.getcwd()

        # Get the device for the current working directory using 'df'.
        # The second line of output is the actual device.
        df_cmd = f"df --output=source {cwd}"
        df_output = subprocess.check_output(df_cmd, shell=True, text=True).strip().splitlines()
        if len(df_output) < 2:
            raise ValueError("Could not determine device from df output")
        current_device = df_output[1].strip()

        # get name, i.e. /dev/sda2 => sda
        if current_device.startswith("/dev"):
            current_device = current_device[5:-1]

        # List only physical disks (not partitions) with fields: NAME, TYPE, SIZE, MODEL.
        lsblk_cmd = "lsblk -d -o NAME,TYPE,SIZE,MODEL -n"
        lsblk_output = subprocess.check_output(lsblk_cmd, shell=True, text=True).strip()
        disk_info_list = []

        for line in lsblk_output.splitlines():
            parts = line.split()

            if len(parts) < 4:
                continue

            name = parts[0]

            # match device name, account for multiple just in case
            if name != current_device:
                continue

            dev_type = parts[1]
            size = parts[2]
            model = " ".join(parts[3:])
            if dev_type != "disk":
                continue

            disk_info_list.append({
                "size": size,
                "model": model
            })

        if disk_info_list:
            disk_capacities = ", ".join(disk["size"] for disk in disk_info_list)
            disk_models = ", ".join(disk["model"] for disk in disk_info_list)
        else:
            disk_capacities = disk_models = "N/A"
    except Exception as e:
        disk_capacities = disk_models = "N/A"

    return {
        "Disk Capacity": disk_capacities,
        "Disk Model": disk_models,
    }


def run_dd_speed_test(test_file="dd_test_file", block_size="1M", count=1024):
    """
    Uses dd to measure disk write and read speeds.
    Returns a tuple: (write_speed, read_speed)
    """
    # Write Speed Test
    try:
        write_cmd = f"dd if=/dev/zero of={test_file} bs={block_size} count={count} oflag=direct"
        write_result = subprocess.run(write_cmd, shell=True, capture_output=True, text=True)
        write_output = write_result.stderr
        write_speed = write_output.strip().splitlines()[-1].split(", ")[-1]
    except Exception as e:
        print(str(e))
        write_speed = "N/A"

    # Read Speed Test
    try:
        read_cmd = f"dd if={test_file} of=/dev/null bs={block_size} count={count} iflag=direct"
        read_result = subprocess.run(read_cmd, shell=True, capture_output=True, text=True)
        read_output = read_result.stderr
        read_speed = read_output.strip().splitlines()[-1].split(", ")[-1]
    except Exception as e:
        read_speed = "N/A"

    # Clean up the test file
    try:
        os.remove(test_file)
    except Exception:
        pass

    return write_speed, read_speed


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
        gpus_str = ",".join(str(i) for i in indices)
    else:
        # Fallback: use all available info if no indices found
        vram = " ".join(all_vram)
        gpu_name = " ".join(all_gpu_names)
        gpus_str = ""

    # Create safe strings for filenames
    gpu_name_safe = gpu_name.replace(" ", "_").replace(",", "-")
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



    # Step 5: Run benchmark (this will create files in the current working directory)
    StopWatch.start("benchmark")
    subprocess.call(
        f"{darknetloc} detector -map -dont_show -verbose -nocolor -gpus {gpus_str} train /workspace/LegoGears_v2/LegoGears.data /workspace/LegoGears_v2/LegoGears.cfg 2>&1 | tee training_output.log",
        shell=True)
    StopWatch.stop("benchmark")
    benchmark_result = StopWatch.get_benchmark()

    # Step 6: Extract sysinfo and benchmark results
    sysinfo = benchmark_result["sysinfo"]
    benchmark = benchmark_result["benchmark"]["benchmark"]

    # Now that we have sysinfo, create a safe CPU name string.
    cpu_name_safe = sysinfo["cpu"].replace(" ", "_")

    # Get disk information using lsblk.
    print("Getting disk information")
    disk_info = get_disk_info()

    # Run dd speed tests to get file-based write and read speeds.
    print("Running disk speed test")
    dd_write_speed, dd_read_speed = run_dd_speed_test()

    data = {
        "Benchmark Time (s)": benchmark["time"],
        "CPU Name": sysinfo["cpu"],
        "CPU Threads": sysinfo["cpu_threads"],
        "GPU Name": gpu_name,
        "GPU VRAM": vram,
        "Total Memory": sysinfo["mem.total"],
        "OS": sysinfo["uname.system"],
        "Architecture": sysinfo["uname.machine"],
        "Python Version": sysinfo["python.version"],
        "Disk Capacity": disk_info["Disk Capacity"],
        "Disk Model": disk_info["Disk Model"],
        "Write Speed": dd_write_speed,
        "Read Speed": dd_read_speed,
    }

    # Step 7: Create a unique CSV filename in the current (output) directory
    filename = f"benchmark__{username}__{gpu_name_safe}__{cpu_name_safe}__{now}.csv"
    with open(filename, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())
        writer.writeheader()
        writer.writerow(data)

    print(f"Benchmark results saved to {filename}")

    
    # Move all files matching "*weights" from /workspace/LegoGears_v2/ to /outputs
    for file in glob.glob("/workspace/LegoGears_v2/*weights"):
        shutil.move(file, output_dir)
