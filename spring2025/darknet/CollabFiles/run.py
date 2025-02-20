import csv
from cloudmesh.common.StopWatch import StopWatch
from cloudmesh.gpu.gpu import Gpu
import subprocess
import train_setup # links training data to its labels

if __name__ == "__main__":
    # Get GPU name
    gpu = Gpu()
    vram = ' '.join(card['fb_memory_usage']['total'] for card in gpu._smi)
    gpu_name = ' '.join(card["product_name"] for card in gpu.system())

    # Run benchmark
    StopWatch.start("benchmark")
    subprocess.call("darknet detector -map -dont_show -verbose -nocolor train /workspace/unzips/cars.data /workspace/unzips/cars.cfg 2>&1 | tee training_output.log", shell=True)
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

    # Write to CSV
    with open("benchmark.csv", mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())
        writer.writeheader()
        writer.writerow(data)

    print("Benchmark results saved to benchmark.csv")
