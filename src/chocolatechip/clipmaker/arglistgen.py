# cd /mnt/hdd/data/video_pipeline/tracking
# find "$(pwd)" -type f -ctime -9 \( -path "*tracking/26_*" -o -path "*tracking/25_*" \)

import sys

import subprocess
import re
from datetime import datetime
import argparse

def get_recent_files(base_path, camera_id, year, days):
    """Runs the find command to get files modified within the last 'days' days for a specific camera ID and year."""
    command = [
        "find", base_path, "-type", "f", "-ctime", f"-{days}",
        "-path", f"*tracking/{camera_id}_{year}*"
    ]

    try:
        # Execute the command and capture output
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return result.stdout.strip().split("\n")
    except subprocess.CalledProcessError as e:
        print(f"Error running find command: {e.stderr}")
        return []

def extract_timestamps(file_paths):
    """Extracts unique dates with their earliest and latest times from file paths."""
    date_times = {}

    for path in file_paths:
        match = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})", path)
        if match:
            date_str, time_str = match.groups()
            time_str = time_str.replace("-", ":")  # Convert 18-45-00 to 18:45:00
            full_datetime = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")

            if date_str not in date_times:
                date_times[date_str] = {"earliest": full_datetime, "latest": full_datetime}
            else:
                date_times[date_str]["earliest"] = min(date_times[date_str]["earliest"], full_datetime)
                date_times[date_str]["latest"] = max(date_times[date_str]["latest"], full_datetime)

    return date_times

def generate_time_intervals(date_times):
    arguments = []
    for date_str, times in date_times.items():
        start_time = times["earliest"].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        end_time   = times["latest"].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        # → no extra quotes here:
        arguments.append(start_time)
        arguments.append(end_time)
    return arguments


if __name__ == "__main__":

    # ✅ Argument Parser for Dynamic Inputs
    parser = argparse.ArgumentParser(
        description="Find video tracking files and generate time intervals."
    )
    parser.add_argument("--camera_id", required=True, help="Camera ID (e.g., 07)")
    parser.add_argument("--year",      required=True, help="Year of files to search")
    parser.add_argument("--days",      type=int, default=30,
                        help="Look back this many days (default: 30)")
    parser.add_argument("--base_path", default="/mnt/hdd/data/video_pipeline/tracking",
                        help="Base directory path")
    args = parser.parse_args()

    # Run the file‑finder
    file_paths = get_recent_files(
        args.base_path, args.camera_id, args.year, args.days
    )

    if not file_paths:
        print("No matching files found.")
        sys.exit(0)

    # Extract earliest/latest per date, then build timestamp list
    date_times = extract_timestamps(file_paths)
    timestamps = generate_time_intervals(date_times)

    # Show what we generated
    print("Generated Argument List:")
    for ts in timestamps:
        print("  ", ts)

    # Write one timestamp per line (no extra quotes)
    with open("arglist.txt", "w") as file:
        for ts in timestamps:
            file.write(f"{ts}\n")
