import csv
from cloudmesh.common.StopWatch import StopWatch
from cloudmesh.gpu.gpu import Gpu
import subprocess
import train_setup # links training data to its labels
import os
import getpass
from datetime import datetime

if __name__ == "__main__":
    # Get GPU name
    gpu = Gpu()
    vram = ' '.join(card['fb_memory_usage']['total'] for card in gpu._smi)
    gpu_name = ' '.join(card["product_name"] for card in gpu.system())

    # gpu.watch(logfile=f'{gpu_name}.log')

    # check environment variable APPTAINER
    if "APPTAINER_ENVIRONMENT" in os.environ:
        print("Running in Apptainer environment")
        darknetloc = '/host_workspace/darknet/build/src-cli/darknet'
        # Set up any necessary environment variables or configurations here
    else:
        print("Running non-apptainer")
        darknetloc = 'darknet'

    # Run benchmark
    StopWatch.start("benchmark")
    subprocess.call(f"{darknetloc} detector -map -dont_show -verbose -nocolor train /workspace/LegoGears_v2/LegoGears.data /workspace/LegoGears_v2/LegoGears.cfg 2>&1 | tee training_output.log", shell=True)
    StopWatch.stop("benchmark")
    benchmark_result = StopWatch.get_benchmark()

    # Extract relevant data
    sysinfo = benchmark_result["sysinfo"]
    benchmark = benchmark_result["benchmark"]["benchmark"]

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

    # Create a unique filename using username, GPU name, CPU name, and current date/time.
    username = getpass.getuser()
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Replace spaces with underscores to make a safe filename
    gpu_name_safe = gpu_name.replace(" ", "_")
    cpu_name_safe = sysinfo["cpu"].replace(" ", "_")
    filename = f"benchmark_{username}_{gpu_name_safe}_{cpu_name_safe}_{now}.csv"

    # Write to CSV with the unique filename
    with open(filename, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())
        writer.writeheader()
        writer.writerow(data)

    print(f"Benchmark results saved to {filename}")