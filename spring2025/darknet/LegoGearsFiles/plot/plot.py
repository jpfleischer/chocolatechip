import os
import csv
import matplotlib.pyplot as plt

def main():
    data = []  # list of tuples: (label, benchmark_time)

    # Walk through ../outputs directory (including subdirectories)
    # Walk through ../outputs directory (including subdirectories)
    # Walk through ../outputs directory (including subdirectories)
    output_dir = os.path.join("..", "outputs")
    for root, dirs, files in os.walk(output_dir):
        for filename in files:
            if not filename.endswith(".csv"):
                continue
            # Full path to the file.
            filepath = os.path.join(root, filename)
            
            # Split the filename by '__'
            # Expected format:
            #  benchmark__denis__NVIDIA_GeForce_RTX_2060__Intel(R)_Core(TM)_i7-9700K_CPU_@_3.60GHz__20250328_064036.csv
            parts = filename.split("__")
            if len(parts) < 4:
                # Skip unexpected file names.
                continue
            
            # Extract GPU and CPU parts.
            # Replace underscores with spaces so that e.g. "NVIDIA_GeForce_RTX_2060" becomes "NVIDIA GeForce RTX 2060"
            gpu_name = parts[2].replace("_", " ")
            cpu_name = parts[3].replace("_", " ")
            label = f"{gpu_name}\n{cpu_name}"
            
            # Open the CSV file and extract the benchmark time from the second row, first column.
            try:
                with open(filepath, "r", newline="") as csvfile:
                    reader = csv.reader(csvfile)
                    header = next(reader)  # Skip header line.
                    row = next(reader, None)
                    if row is not None:
                        # Try to convert the first column to a float.
                        benchmark_time = float(row[0])
                    else:
                        print(f"Warning: File {filepath} contains no data row.")
                        continue
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                continue

            # Skip benchmarks with time less than 10 seconds.
            if benchmark_time < 10:
                continue
                
            # Save the (label, benchmark_time) tuple.
            data.append((label, benchmark_time))

    if not data:
        print("No benchmark data found!")
        return

    # Sort the data by benchmark (training) time, largest first.
    data = sorted(data, key=lambda x: x[1], reverse=True)
    labels, times = zip(*data)

    # Create a bar chart with a wider figure.
    fig, ax = plt.subplots(figsize=(15, 6))
    x_positions = range(len(labels))
    ax.bar(x_positions, times, color="skyblue")

    # Set x-axis tick labels.
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    ax.set_ylabel("Benchmark Time (s)")
    ax.set_title("Benchmark Results by GPU and CPU (Ranked by Training Time)")

    plt.tight_layout()
    plt.savefig("myplot.pdf",bbox_inches = "tight")
    plt.show()
if __name__ == "__main__":
    main()