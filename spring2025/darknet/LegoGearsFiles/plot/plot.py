import os
import csv
import matplotlib.pyplot as plt
from datetime import datetime
import itertools
from matplotlib.patches import Patch
from username_machine_pairings import get_machine  # import your get_machine function
import numpy as np
from scipy import signal

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

def plot_benchmark(toggle_multiple_gpus=False):
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
            if not MULTIPLE_GPUS:
                if not toggle_multiple_gpus and is_multiple_gpu(gpu_name):
                    continue
                elif toggle_multiple_gpus and not is_multiple_gpu(gpu_name):
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

def plot_benchmark_with_str(filter_string = "NVIDIA RTX A6000"):
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

            if filter_string not in gpu_name:
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
            machine = 'Single GPU' if '-' not in gpu_name else 'Dual GPU'
            benchmark_data.append((label, benchmark_time, machine))

    if not benchmark_data:
        print("No benchmark data found!")
        return

    # Sort benchmark data by training time (smallest first)
    benchmark_data = sorted(benchmark_data, key=lambda x: x[1])
    labels, times, machines = zip(*benchmark_data)

    # Use tab10 colormap to assign colors per machine.
    unique_machines = sorted(set(machines))
    cmap = plt.get_cmap("tab10")
    machine_colors = {machine: cmap(i) for i, machine in enumerate(unique_machines)}
    # Create a list of bar colors according to each bar's machine.
    bar_colors = [machine_colors[machine] for machine in machines]

    n = len(labels)
    # Use np.linspace to get positions evenly spread from 0 to 1.
    x_positions = np.linspace(0, 1, n)
    # Define the bar width as a fraction of the spacing between bars.
    if n > 1:
        spacing = x_positions[1] - x_positions[0]
    else:
        spacing = 0.2  # arbitrary for a single bar
    bar_width = spacing * 0.4  # adjust this multiplier to make bars thicker/thinner

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x_positions, times, width=bar_width, color=bar_colors, align='center')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Benchmark Time (s)")

    # Create legend elements for each machine.
    legend_elements = [Patch(facecolor=machine_colors[machine], label=machine) for machine in unique_machines]
    ax.legend(handles=legend_elements, title="Machine", loc="upper right")

    # Set the limits so that there is equal space on the sides.
    ax.set_xlim(-spacing * 0.5, 1 + spacing * 0.5)

    plt.tight_layout()
    plt.savefig("dual_gpu_training_times.pdf", bbox_inches="tight")
    plt.show()


def plot_gpu_data(csv_col_name='gpu_temp C', y_axis="Temperature", units="°C", bin_range=None):
    """
    Reads GPU data logs and creates a plot of the data over time.
    
    If bin_range (in seconds) is provided, the temperature readings are averaged within
    each bin before plotting.

    The function now also trims the raw data by:
      1. Removing the last 20 seconds.
      2. Removing an initial segment so that the remaining duration equals the benchmark time,
         as read from a CSV file in the folder.
    
    Returns the plot figure for further customization if needed.
    """

    gpu_runs = {}  # Map of gpu_label -> [(times, temps, machine_type)]
    output_dir = os.path.join("..", "outputs")

    for root, _, files in os.walk(output_dir):
        if "mylogfile.log" not in files:
            continue

        # Identify CSV files in the folder (could include the benchmark CSV)
        csv_files = [f for f in files if f.endswith(".csv")]
        if not csv_files:
            continue

        # Expected filename format: benchmark__<username>__NVIDIA_GeForce_RTX...
        gpu_label = None
        machine_type = "Personal Machine"
        for csv_file in csv_files:
            parts = csv_file.split("__")
            if len(parts) < 3:
                continue

            username = parts[1]  # Extract username correctly from filename
            gpu_label = parts[2].replace("_", " ")
            # Determine machine type based on username
            machine_type = get_machine(username)
            break

        if not gpu_label:
            gpu_label = "Unknown GPU"

        log_path = os.path.join(root, "mylogfile.log")
        try:
            with open(log_path, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                temp_cols = [col for col in reader.fieldnames if csv_col_name in col]
                if not temp_cols:
                    print(f"No {csv_col_name} columns found in {log_path}")
                    continue

                timestamps, avg_temps = [], []
                for row in reader:
                    try:
                        # Convert the first column to a datetime object
                        timestamp = datetime.fromisoformat(row[reader.fieldnames[0]])
                        # Gather valid temperature readings
                        temps = []
                        for col in temp_cols:
                            val = row.get(col, "").strip()
                            if val:
                                try:
                                    temps.append(float(val))
                                except ValueError:
                                    continue
                        if temps:
                            timestamps.append(timestamp)
                            avg_temps.append(sum(temps) / len(temps))
                    except Exception:
                        continue

                if timestamps and avg_temps:
                    base_time = timestamps[0]
                    elapsed = [(t - base_time).total_seconds() for t in timestamps]

                    # ----- TRIMMING THE DATA BASED ON BENCHMARK TIME -----
                    # Remove the last 20 seconds from the run.
                    total_elapsed = elapsed[-1]
                    trimmed_end = total_elapsed - 20

                    # Attempt to read the benchmark time from one of the CSV files.
                    # The benchmark CSV is assumed to have a header "Benchmark Time (s)".
                    benchmark_time = None
                    for f in csv_files:
                        fpath = os.path.join(root, f)
                        try:
                            with open(fpath, "r", newline="") as fcsv:
                                csv_reader = csv.reader(fcsv)
                                header = next(csv_reader, None)
                                if header and header[0].strip() == "Benchmark Time (s)":
                                    row = next(csv_reader, None)
                                    if row:
                                        benchmark_time = float(row[0])
                                        break
                        except Exception:
                            continue

                    if benchmark_time is None:
                        print(f"Benchmark time not found in {root}, skipping trimming for this run.")
                    else:
                        # We now want the remaining data (after chopping off the final 20 sec)
                        # to span exactly the benchmark time.
                        # Calculate the extra time at the beginning to drop:
                        desired_duration = benchmark_time
                        offset = max(trimmed_end - desired_duration, 0)
                        # Filter the data to include only points between [offset, trimmed_end]
                        new_elapsed = []
                        new_temps = []
                        for t, temp in zip(elapsed, avg_temps):
                            if offset <= t <= trimmed_end:
                                new_elapsed.append(t - offset)  # rebase time to zero
                                new_temps.append(temp)
                        elapsed = new_elapsed
                        avg_temps = new_temps
                    # ----- END TRIMMING -----

                    # Store the machine type along with the data
                    gpu_runs.setdefault(gpu_label, []).append((elapsed, avg_temps, machine_type))
        except Exception as e:
            print(f"Error processing {log_path}: {e}")
            continue

    if not gpu_runs:
        print("No data to plot")
        return None

    # Set up the main plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Define markers for different GPUs
    markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'X', '+', 'x']

    # Define pastel colors for machine types
    machine_colors = {
        "Supercomputer": "#6495ED",  # Pastel blue
        "Personal Machine": "#FF6961"  # Pastel red
    }

    # Assign a unique marker to each unique GPU model
    unique_gpus = list(gpu_runs.keys())
    gpu_markers = {gpu: markers[i % len(markers)] for i, gpu in enumerate(unique_gpus)}

    # Keep track of which combinations of (gpu, machine_type) we've seen
    # to avoid duplicate legend entries
    legend_entries = set()

    for gpu, runs in gpu_runs.items():
        # Get marker for this GPU
        marker = gpu_markers[gpu]

        for elapsed, temps, machine_type in runs:
            # Determine if it's a supercomputer
            is_supercomputer = "HiPerGator" in machine_type or "Afton" in machine_type
            machine_category = "Supercomputer" if is_supercomputer else "Personal Machine"
            color = machine_colors[machine_category]

            # For the label, combine GPU model and machine type
            legend_key = (gpu, machine_category)
            if legend_key not in legend_entries:
                label = f"{gpu}"
                legend_entries.add(legend_key)
            else:
                label = None  # Don't add duplicate legend entries

            if bin_range is not None and bin_range > 0:
                binned_times = {}
                for t, temp in zip(elapsed, temps):
                    bin_key = int(t // bin_range)
                    binned_times.setdefault(bin_key, []).append((t, temp))

                # Average the data in each bin
                binned_x = []
                binned_y = []
                for key in sorted(binned_times.keys()):
                    group = binned_times[key]
                    times_in_bin, temps_in_bin = zip(*group)
                    avg_time = np.mean(times_in_bin)
                    avg_temp = np.mean(temps_in_bin)
                    binned_x.append(avg_time)
                    binned_y.append(avg_temp)

                ax.plot(binned_x, binned_y, label=label, 
                        color=color, linestyle='-', alpha=0.8,
                        marker=marker, markersize=6, markevery=max(1, len(binned_x)//20))
            else:
                ax.plot(elapsed, temps, label=label, 
                        color=color, linestyle='-', alpha=0.8,
                        marker=marker, markersize=6, markevery=max(1, len(elapsed)//20))

    ax.set(xlabel='Time (seconds)', ylabel=f'{y_axis} ({units})', title=f'GPU {y_axis} Over Time')
    ax.grid(True, linestyle='--', alpha=0.7)

    if len(legend_entries) > 5:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    else:
        ax.legend()

    for gpu, runs in gpu_runs.items():
        for machine_category in ["Supercomputer", "Personal Machine"]:
            category_temps = []
            for _, temps, machine_type in runs:
                is_supercomputer = "HiPerGator" in machine_type or "Afton" in machine_type
                if (machine_category == "Supercomputer" and is_supercomputer) or \
                   (machine_category == "Personal Machine" and not is_supercomputer):
                    category_temps.extend(temps)

            if category_temps:
                print(f"Average {y_axis} for {gpu} on {machine_category}: {np.mean(category_temps):.1f}{units}")

    plt.tight_layout()

    # --- Add Inset for Machine Type Swatches ---
    inset_ax = fig.add_axes([0.75, 0.75, 0.2, 0.2])
    inset_ax.axis('off')
    import matplotlib.patches as mpatches
    personal_patch = mpatches.Patch(color=machine_colors["Personal Machine"], label="Personal Machine")
    super_patch = mpatches.Patch(color=machine_colors["Supercomputer"], label="Supercomputer")
    inset_ax.legend(handles=[personal_patch, super_patch], loc='center', frameon=False)
    inset_ax.set_title("Machine Types", fontsize=10)

    pdf_filename = f'{y_axis.lower()}_plot.pdf'
    plt.savefig(pdf_filename, bbox_inches='tight')
    print(f"Plot saved to {pdf_filename}")

    plt.show()

    return fig


def plot_gpu_before_after(csv_col_name='gpu_temp C', y_axis="Temperature", units="°C", window_size=7, file_name="temp"):
    """
    Reads GPU data logs and creates side-by-side bar charts for each GPU showing:
      - Average before training 
      - Average during training
      - Average after cooling
    
    Uses improved phase detection to more accurately identify temperature phases.
    Includes machine type (Personal Machine, UVA, HiPerGator) in GPU labels.
    
    Returns the bar chart figure.
    """
    # Dictionary to accumulate temperatures per GPU
    gpu_data = {}

    output_dir = os.path.join("..", "outputs")
    for root, _, files in os.walk(output_dir):
        if "mylogfile.log" not in files:
            continue
            
        # Try to find username from the directory path
        username = None
        for file in files:
            if file.endswith(".csv") and "__" in file:
                parts = file.split("__")
                if len(parts) >= 2:
                    username = parts[1]
                    break
                    
        if not username:
            username = "unknown"
            
        # Determine machine type using the existing get_machine function
        machine_type = get_machine(username)

        # Determine GPU label from CSV filename with expected format
        gpu_label = next(
            (f.split("__")[2].replace("_", " ") for f in files
             if f.endswith(".csv") and "__" in f and len(f.split("__")) >= 3),
            "Unknown GPU"
        )
        
        # Add machine type to the GPU label
        gpu_label = f"{gpu_label}\n{machine_type}"

        log_path = os.path.join(root, "mylogfile.log")
        try:
            with open(log_path, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                temp_cols = [col for col in reader.fieldnames if csv_col_name in col]
                if not temp_cols:
                    print(f"No {csv_col_name} columns found in {log_path}")
                    continue

                # For each run in the log file, we collect the temperature values
                run_temps = []  # list of temperature sequences for this run
                temps_this_run = []
                # Assume each row is part of one continuous run.
                for row in reader:
                    try:
                        # Convert the first column to datetime
                        _ = datetime.fromisoformat(row[reader.fieldnames[0]])
                        # Get average temperature from the desired columns
                        temps = []
                        for col in temp_cols:
                            val = row.get(col, "").strip()
                            if val:
                                try:
                                    temps.append(float(val))
                                except ValueError:
                                    continue
                        if temps:
                            # use the average for this row
                            temps_this_run.append(sum(temps)/len(temps))
                    except Exception:
                        continue
                if temps_this_run:
                    run_temps.append(temps_this_run)

                # Process each run with improved phase detection
                for temps in run_temps:
                    if len(temps) < window_size*3:  # Need enough data points
                        continue
                    
                    # Apply a moving average to smooth out noise
                    def moving_average(data, window_size):
                        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
                    
                    # Get smoothed temperature curve
                    smoothed_temps = moving_average(temps, window_size)
                    
                    # IMPROVED PHASE DETECTION:
                    # 1. Find baseline temperature (before training)
                    # 2. Find peak temperature (during training)
                    # 3. Find cooldown period (after training)

                    # Detect significant temperature increase
                    temp_range = max(smoothed_temps) - min(smoothed_temps)
                    if temp_range < 5:  # Not enough temperature variation
                        # Just use simple thirds
                        third = len(temps) // 3
                        before = [temps[0]]
                        during = temps[third:2*third]
                        after = temps[2*third:]
                    else:
                        # Calculate rate of change
                        gradient = np.gradient(smoothed_temps)
                        
                        # Find when temperature starts to rise significantly
                        baseline_temp = np.mean(smoothed_temps[:window_size])
                        sig_rise_threshold = np.std(gradient) * 1.5
                        
                        # Find rising edge (start of training)
                        rise_points = [i for i, g in enumerate(gradient) if g > sig_rise_threshold]
                        
                        # Find falling edge (end of training/start of cooling)
                        fall_points = [i for i, g in enumerate(gradient) if g < -sig_rise_threshold]
                        
                        if rise_points and fall_points and max(rise_points) < max(fall_points):
                            # We have a clear rise and fall pattern
                            rise_idx = rise_points[0]
                            fall_idx = max([p for p in fall_points if p > rise_idx])
                            
                            # Adjust for the window_size offset
                            rise_idx = max(0, rise_idx + window_size//2)
                            fall_idx = min(len(temps)-1, fall_idx + window_size//2)
                            
                            # Add padding to catch the full profile
                            before = [temps[0]]
                            during = temps[rise_idx:fall_idx]
                            after = temps[fall_idx:]
                        else:
                            # Use temperature thresholds for segmentation
                            # Find the peak temperature
                            peak_temp = max(smoothed_temps)
                            peak_idx = np.argmax(smoothed_temps) + window_size//2
                            
                            # Define temperature thresholds
                            rise_threshold = baseline_temp + 0.2 * (peak_temp - baseline_temp)
                            fall_threshold = baseline_temp + 0.5 * (peak_temp - baseline_temp)
                            
                            # Find the first point where temperature exceeds rise_threshold
                            rise_idx = next((i for i, t in enumerate(temps) if t > rise_threshold), len(temps)//4)
                            
                            # Find points after peak where temperature falls below fall_threshold
                            if peak_idx < len(temps) - 1:
                                fall_candidates = [i for i in range(peak_idx, len(temps)) if temps[i] < fall_threshold]
                                fall_idx = min(fall_candidates) if fall_candidates else len(temps) - 1
                            else:
                                fall_idx = len(temps) - 1
                            
                            before = [temps[0]]
                            during = temps[rise_idx:fall_idx]
                            after = temps[fall_idx:]
                        
                        # Sanity check: make sure phases follow expected temperature pattern
                        if before and during and len(before) > 5 and len(during) > 5:
                            before_avg = np.mean(before)
                            during_avg = np.mean(during)
                            
                            # If "during" isn't warmer than "before" by at least 3°C, something is wrong
                            if during_avg - before_avg < 3.0:
                                # Try finding a significant temperature jump
                                temp_jumps = [(i, temps[i+5] - temps[i]) 
                                              for i in range(len(temps)-5) 
                                              if temps[i+5] - temps[i] > 5.0]
                                
                                if temp_jumps:
                                    # Use the largest jump as the dividing point
                                    temp_jumps.sort(key=lambda x: x[1], reverse=True)
                                    rise_idx = temp_jumps[0][0]
                                    
                                    # Find where temperature drops after the peak
                                    peak_idx = np.argmax(temps[rise_idx:]) + rise_idx
                                    post_peak = temps[peak_idx:]
                                    
                                    if len(post_peak) > 10:
                                        # Look for significant temperature drop after peak
                                        temp_drops = [(i+peak_idx, post_peak[0] - post_peak[i])
                                                     for i in range(1, len(post_peak))
                                                     if post_peak[0] - post_peak[i] > 5.0]
                                        
                                        if temp_drops:
                                            temp_drops.sort(key=lambda x: x[1], reverse=True)
                                            fall_idx = temp_drops[0][0]
                                        else:
                                            fall_idx = len(temps) - 1
                                    else:
                                        fall_idx = len(temps) - 1
                                    
                                    before = [temps[0]]
                                    during = temps[rise_idx:fall_idx]
                                    after = temps[fall_idx:]
                                else:
                                    # Fall back to simple percentile-based segmentation
                                    sorted_temps = sorted(temps)
                                    lower_bound = np.percentile(sorted_temps, 25)
                                    upper_bound = np.percentile(sorted_temps, 75)
                                    
                                    before = [temps[0]]
                                    during = [t for t in temps if t >= upper_bound - 3]
                                    after = [t for t in temps[2*len(temps)//3:] if lower_bound <= t <= upper_bound]
                    
                    # Save data for this GPU
                    if gpu_label not in gpu_data:
                        gpu_data[gpu_label] = {"before": [], "during": [], "after": []}
                    
                    if before:
                        gpu_data[gpu_label]["before"].extend(before)
                    if during:
                        gpu_data[gpu_label]["during"].extend(during)
                    if after:
                        gpu_data[gpu_label]["after"].extend(after)
                    
        except Exception as e:
            print(f"Error processing {log_path}: {e}")
            continue

    if not gpu_data:
        print("No data to plot")
        return None
    
    # Calculate average temperatures for each GPU for before, during, and after training
    gpu_labels = []
    before_avgs = []
    during_avgs = []
    after_avgs = []

    # Sort GPUs by machine type and then by name
    sorted_gpus = sorted(gpu_data.keys(), key=lambda x: (x.split('(')[-1], x))

    for gpu in sorted_gpus:
        data = gpu_data[gpu]
        gpu_labels.append(gpu)
        before_avgs.append(np.mean(data["before"]) if data["before"] else np.nan)
        during_avgs.append(np.mean(data["during"]) if data["during"] else np.nan)
        after_avgs.append(np.mean(data["after"]) if data["after"] else np.nan)
        
        # Final sanity check for temperature relationships
        if (not np.isnan(before_avgs[-1])) and (not np.isnan(during_avgs[-1])):
            if before_avgs[-1] >= during_avgs[-1]:
                # Swap if "before" temps are higher than "during"
                temp = before_avgs[-1]
                before_avgs[-1] = during_avgs[-1]
                during_avgs[-1] = temp
                print(f"Warning: Had to swap before/during temperatures for {gpu}")
        
        # Ensure "after" temps are between "before" and "during" if all three exist
        if (not np.isnan(after_avgs[-1])) and (not np.isnan(before_avgs[-1])) and (not np.isnan(during_avgs[-1])):
            if after_avgs[-1] >= during_avgs[-1]:
                after_avgs[-1] = (before_avgs[-1] + during_avgs[-1]) / 2
                print(f"Warning: Had to adjust after temperatures for {gpu}")

    # --- Plotting ---
    # Create grouped bar chart with 2 groups (Before and During)
    x = np.arange(len(gpu_labels))
    width = 0.4  # Bar width

    # Increase the figure size for additional clarity
    fig, ax = plt.subplots(figsize=(20, 8))  

    rects1 = ax.bar(x - width/2, before_avgs, width, label='Before Training', color='skyblue')
    rects2 = ax.bar(x + width/2, during_avgs, width, label='During Training', color='salmon')

    # Increase font sizes for labels and title
    ax.set_ylabel(f'{y_axis} ({units})', fontsize=16)
    ax.set_title(f'Average {y_axis} Before and During Training', fontsize=18)

    ax.set_xticks(x)
    # Rotate x-axis labels 45° for readability while using a larger font size
    ax.set_xticklabels(gpu_labels, rotation=45, ha='right', rotation_mode='anchor')
    ax.tick_params(axis='x', labelsize=14)  # Larger x-axis label text

    ax.legend(fontsize=14)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Annotate bars with their height values (larger font for annotations)
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if not np.isnan(height):
                ax.annotate(f'{height:.1f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=12)

    autolabel(rects1)
    autolabel(rects2)

    max_val = max(val for val in (before_avgs + during_avgs) if not np.isnan(val))
    ax.set_ylim(top=max_val + 20)  # Add a buffer above your highest bar

    # Optionally adjust y-ticks (for example, every 5 degrees):
    # y_ticks = np.arange(0, max_val + 10, 10)
    # ax.set_yticks(y_ticks)

    # Adjust the subplot parameters to give more space for rotated labels
    plt.subplots_adjust(bottom=0.35, top=0.85)  # Increase top margin to avoid clipping
    plt.tight_layout()
    plt.savefig(f"gpu_{file_name}_before_during.pdf", bbox_inches="tight")
    plt.show()
    return fig

def main():
    plot_benchmark_with_str()
    # plot_gpu_data(bin_range=5)
    # plot_gpu_data("gpu_util %", "Utilization", "%")
    # plot_gpu_data("power_draw W", "Power Draw", "W", bin_range=5)
    # plot_gpu_before_after()
    # plot_gpu_before_after("power_draw W", "Power Draw", "W", file_name="power")

if __name__ == "__main__":
    main()
