import csv
from cloudmesh.common.StopWatch import StopWatch
from cloudmesh.gpu.gpu import Gpu
import subprocess
import os
import getpass
from datetime import datetime
# import train_setup # links training data to its labels
import glob
import shutil


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
        gpus_str = ",".join(str(i) for i in range(len(indices)))
    else:
        # Fallback: use all available info if no indices found
        vram = " ".join(all_vram)
        gpu_name = " ".join(all_gpu_names)
        gpus_str = ""

    # Create safe strings for filenames
    gpu_name_safe = gpu_name.replace(" ", "_").replace(",", "-")
    # We'll set cpu_name_safe later after we obtain sysinfo

    if "APPTAINER_ENVIRONMENT" in os.environ:
        print("Running in Apptainer environment")
        output_dir = os.path.join("/workspace/outputs", f"benchmark__{username}__{gpu_name_safe}__{now}")
        os.makedirs(output_dir, exist_ok=True)
        os.chdir(output_dir)
        print(f"Changed directory to output folder: {output_dir}")
        darknetloc = '/host_workspace/darknet/build/src-cli/darknet'
    else:
        output_dir = os.getcwd()
        print("Running non-apptainer")
        darknetloc = 'darknet'

    # Run benchmark
    StopWatch.start("benchmark")
    command = f"{darknetloc} detector -map -dont_show -nocolor -gpus {gpus_str} train /workspace/cars.data /workspace/cars.cfg 2>&1 | tee training_output.log"
    print("#" * 80)
    print(command)
    print("#" * 80)
    subprocess.call(command, shell=True)

    StopWatch.stop("benchmark")
    benchmark_result = StopWatch.get_benchmark()

    # Extract relevant data
    sysinfo = benchmark_result["sysinfo"]
    benchmark = benchmark_result["benchmark"]["benchmark"]

    # Now that we have sysinfo, create a safe CPU name string.
    cpu_name_safe = sysinfo["cpu"].replace(" ", "_")

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
    }

    # Write to CSV
    filename = f"benchmark__{username}__{gpu_name_safe}__{cpu_name_safe}__{now}.csv"
    with open(filename, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())
        writer.writeheader()
        writer.writerow(data)

    print("Benchmark results saved to benchmark.csv")

    for file in glob.glob("/workspace/*weights"):
        shutil.move(file, output_dir)