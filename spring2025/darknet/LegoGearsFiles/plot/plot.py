import os
import csv
import matplotlib.pyplot as plt
from datetime import datetime
import itertools
from matplotlib.patches import Patch
from username_machine_pairings import get_machine  # import your get_machine function

# Flag to control whether to include multiple GPU runs.
MULTIPLE_GPUS = False  # Set to False to filter out runs with multiple GPUs.

def is_multiple_gpu(gpu_name):
    """
    Determine if the GPU name indicates a multiple-GPU run.
    We assume that if the GPU name contains a hyphen and the two parts
    (split by the hyphen) are identical (ignoring whitespace), then it's a multi-GPU run.
    """
    if '-' in gpu_name:
        parts = gpu_name.split('-')
        # Only consider exactly two parts for this heuristic.
        if len(parts) == 2 and parts[0].strip() == parts[1].strip():
            return True
    return False

def plot_benchmark():
    # List of tuples: (label, benchmark_time, machine)
    benchmark_data = []

    output_dir = os.path.join("..", "outputs")
    for root, dirs, files in os.walk(output_dir):
        for filename in files:
            if not filename.endswith(".csv"):
                continue
            filepath = os.path.join(root, filename)
            
            # Expected filename format:
            # benchmark__<username>__NVIDIA_GeForce_RTX_2060__Intel(R)_Core(TM)_i7-9700K_CPU_@_3.60GHz__20250328_064036.csv
            parts = filename.split("__")
            if len(parts) < 4:
                continue

            username = parts[1]  # extract username from the filename
            gpu_name = parts[2].replace("_", " ")
            cpu_name = parts[3].replace("_", " ")
            # Remove clock speed info: chop off "@" and everything after (and trim whitespace).
            if "@" in cpu_name:
                cpu_name = cpu_name.split("@")[0].strip()
            label = f"{gpu_name}\n{cpu_name}"

            # Filter out multiple GPU runs if the flag is set to False.
            if not MULTIPLE_GPUS and is_multiple_gpu(gpu_name):
                continue

            try:
                with open(filepath, "r", newline="") as csvfile:
                    reader = csv.reader(csvfile)
                    next(reader)  # Skip header.
                    row = next(reader, None)
                    if row is not None:
                        benchmark_time = float(row[0])
                    else:
                        print(f"Warning: File {filepath} contains no data row.")
                        continue
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                continue

            if benchmark_time < 10:
                continue

            # Determine the machine using the get_machine function.
            machine = get_machine(username)
            benchmark_data.append((label, benchmark_time, machine))

    if not benchmark_data:
        print("No benchmark data found!")
        return

    # Sort benchmark data by training time (largest first)
    benchmark_data = sorted(benchmark_data, key=lambda x: x[1], reverse=True)
    labels, times, machines = zip(*benchmark_data)

    # Use tab10 colormap to assign colors per machine.
    unique_machines = sorted(set(machines))
    cmap = plt.get_cmap("tab10")
    machine_colors = {machine: cmap(i) for i, machine in enumerate(unique_machines)}
    # Create a list of bar colors according to each bar's machine.
    bar_colors = [machine_colors[machine] for machine in machines]

    fig, ax = plt.subplots(figsize=(15, 6))
    x_positions = range(len(labels))
    ax.bar(x_positions, times, color=bar_colors)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Benchmark Time (s)")
    # ax.set_title("Benchmark Results by GPU and CPU (Ranked by Training Time)")

    # Create legend elements for each machine.
    legend_elements = [Patch(facecolor=machine_colors[machine], label=machine) for machine in unique_machines]
    ax.legend(handles=legend_elements, title="Machine", loc="upper right")

    plt.tight_layout()
    plt.savefig("gpu_training_time.pdf", bbox_inches="tight")
    plt.show()

def plot_gpu_temperature_from_logs():
    # Dictionary mapping GPU name (extracted from a CSV file in the same subdirectory)
    # to a list of runs, where each run is a tuple (list_of_elapsed_seconds, list_of_avg_temperatures)
    gpu_runs = {}
    output_dir = os.path.join("..", "outputs")
    
    # Walk through each subdirectory in outputs
    for root, dirs, files in os.walk(output_dir):
        if "mylogfile.log" not in files:
            continue
        
        # Try to find a CSV file in this directory to extract the GPU name.
        gpu_label = None
        for f in files:
            if f.endswith(".csv") and "__" in f:
                parts = f.split("__")
                if len(parts) >= 3:
                    gpu_label = parts[2].replace("_", " ")
                    break
        if gpu_label is None:
            gpu_label = "Unknown GPU"
        
        log_path = os.path.join(root, "mylogfile.log")
        try:
            with open(log_path, "r") as logfile:
                lines = logfile.readlines()
        except Exception as e:
            print(f"Error reading {log_path}: {e}")
            continue
        if not lines:
            continue
        
        # Skip header (first line starting with "time,")
        data_lines = [line.strip() for line in lines if not line.startswith("time,") and line.strip()]
        if not data_lines:
            continue
        
        run_times = []
        run_avg_temps = []
        for line in data_lines:
            fields = line.split(",")
            if len(fields) < 2:
                continue
            try:
                timestamp = datetime.fromisoformat(fields[0])
            except Exception as e:
                print(f"Error parsing timestamp in line: {line} -> {e}")
                continue
            # Determine the number of GPUs: after the timestamp, each GPU contributes 9 fields.
            num_gpus = (len(fields) - 1) // 9
            temps = []
            for i in range(num_gpus):
                # Each GPU's temperature is at index 6 + 9*i.
                idx = 6 + 9 * i
                try:
                    temp_val = float(fields[idx])
                    temps.append(temp_val)
                except Exception as e:
                    print(f"Error parsing temperature for GPU {i} in line: {line} -> {e}")
                    continue
            if temps:
                avg_temp = sum(temps) / len(temps)
                run_times.append(timestamp)
                run_avg_temps.append(avg_temp)
        if run_times and run_avg_temps:
            # Convert timestamps to elapsed seconds (starting at 0)
            base_time = run_times[0]
            elapsed_seconds = [(t - base_time).total_seconds() for t in run_times]
            if gpu_label not in gpu_runs:
                gpu_runs[gpu_label] = []
            gpu_runs[gpu_label].append((elapsed_seconds, run_avg_temps))
    
    # Prepare to assign a fixed color for each GPU label.
    color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    gpu_colors = {gpu: next(color_cycle) for gpu in gpu_runs.keys()}
    
    # Plot the temperature time series for each GPU.
    plt.figure(figsize=(15, 6))
    for gpu_label, runs in gpu_runs.items():
        color = gpu_colors[gpu_label]
        for idx, (elapsed_times, avg_temps) in enumerate(runs):
            label = gpu_label if idx == 0 else None
            plt.plot(elapsed_times, avg_temps, label=label, color=color)
    plt.xlabel("Elapsed Time (s)")
    plt.ylabel("Average GPU Temperature (°C)")
    # plt.title("GPU Temperature Over Time by GPU")
    plt.legend()
    plt.tight_layout()
    plt.savefig("temperature_plot.pdf", bbox_inches="tight")
    plt.show()

def main():
    plot_benchmark()
    plot_gpu_temperature_from_logs()

if __name__ == "__main__":
    main()
