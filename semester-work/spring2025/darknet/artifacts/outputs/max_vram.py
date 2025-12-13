import csv
import re

def analyze_log(log_file_path):
    """
    Analyzes a benchmark log file to find maximum VRAM usage.

    Args:
        log_file_path (str): The full path to the log file.

    Returns:
        float: The maximum VRAM used in MiB, or 0 if no data is found.
    """
    max_vram = 0
    
    # Regex to extract the numeric part of the memory usage string (e.g., "18386MiB")
    mem_pattern = re.compile(r'(\d+)')

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            # Skip header line
            header_line = next(f, None)
            if not header_line:
                print(f"Warning: Log file '{log_file_path}' is empty.")
                return 0
            
            # Find the index for VRAM usage
            headers = [h.strip() for h in header_line.split(',')]
            try:
                # The column name is ' 0 vram_mem_used MiB' in the log
                vram_col_name = '0 vram_mem_used MiB'
                vram_idx = headers.index(vram_col_name)
            except ValueError:
                print(f"Error: Column '{vram_col_name}' not found in '{log_file_path}'.")
                return 0

            reader = csv.reader(f)
            for row in reader:
                if len(row) > vram_idx:
                    vram_str = row[vram_idx]
                    match = mem_pattern.match(vram_str)
                    if match:
                        vram_used = int(match.group(1))
                        if vram_used > max_vram:
                            max_vram = vram_used
    except FileNotFoundError:
        print(f"Error: File not found at '{log_file_path}'")
        return 0
    except Exception as e:
        print(f"An error occurred while processing '{log_file_path}': {e}")
        return 0

    return max_vram

if __name__ == '__main__':
    # You can add the paths to your log files here
    log_files = [
        '/home/ibraheem.qureshi/clone/chocolatechip/semester-work/spring2025/darknet/artifacts/outputs/LegoGearsDarknet/yolov4/benchmark__ibraheem.qureshi__NVIDIA_B200__darknet__val20__416x416__20251207_235312/mylogfile.log',
    ]

    overall_max_vram = 0
    for log_file in log_files:
        print(f"Analyzing '{log_file}'...")
        current_max = analyze_log(log_file)
        if current_max > 0:
            print(f"Maximum VRAM used in this file: {current_max} MiB")
            if current_max > overall_max_vram:
                overall_max_vram = current_max
        print("-" * 20)

    if overall_max_vram > 0:
        print(f"\nOverall maximum VRAM used across all log files: {overall_max_vram} MiB")

