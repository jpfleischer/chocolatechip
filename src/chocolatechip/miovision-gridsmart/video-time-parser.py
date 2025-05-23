import os
import re
import json
import subprocess
from datetime import datetime, timedelta
import pandas as pd
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

# Use glob to get files matching the specific pattern.
# video_files = glob.glob("/mnt/vast/BrowardVideosAll/2024/2024*68Av*mp4")
video_files = glob.glob("/mnt/huge/GainesvilleVideos/2022/07_2022-10*mp4")
# video_files = glob.glob("/mnt/huge/GainesvilleVideos/2024/oct24/07_2024-10*mp4")


# Regular expression to extract start date and time from filename.
# Example filename: 2024-12-31_09-30-02-rtsp_Stirling-68Av_0.mp4
pattern = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})")

# List to hold tuples of (start_dt, start_str, end_str)
video_times = []

# Variable to accumulate total duration (in seconds)
total_seconds = 0.0

for video_path in tqdm(video_files, desc="Processing videos"):
    filename = os.path.basename(video_path)
    match = pattern.search(filename)
    if not match:
        continue  # Skip files that don't match the expected pattern

    # Parse the start datetime from the filename.
    start_date = match.group(1)              # e.g., "2024-12-31"
    start_time_str = match.group(2).replace("-", ":")  # e.g., "09:30:02"
    try:
        start_dt = datetime.strptime(f"{start_date} {start_time_str}", "%Y-%m-%d %H:%M:%S")
    except Exception as e:
        continue  # Skip if the datetime cannot be parsed

    try:
        # Use ffprobe to extract the video duration (in seconds).
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path],
            capture_output=True, text=True, check=True
        )
        metadata = json.loads(result.stdout)
        duration = float(metadata["format"]["duration"])
        if duration <= 0:
            continue  # Skip if duration is zero or negative
    except Exception as e:
        continue  # Skip broken or unreadable videos

    # Accumulate total duration.
    total_seconds += duration

    # Calculate the end time.
    end_dt = start_dt + timedelta(seconds=duration)

    # Format the datetimes as strings with .000 appended.
    start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S") + ".000"
    end_str = end_dt.strftime("%Y-%m-%d %H:%M:%S") + ".000"

    video_times.append((start_dt, start_str, end_str))

# Sort the list by the actual start datetime.
video_times.sort(key=lambda x: x[0])

# Create a DataFrame for inspection (optional)
df = pd.DataFrame([(start_str, end_str) for _, start_str, end_str in video_times], 
                  columns=["start", "end"])
print("Video Times DataFrame:")
print(df)

# Print total duration information.
print(f"Total duration in seconds: {total_seconds:.2f}")
print(f"Total duration in hours: {total_seconds/3600:.2f}")

# Save the results to a file in the desired format.
output_file = "video_times_output.py"
with open(output_file, "w") as f:
    for _, start_str, end_str in video_times:
        f.write(f"'{start_str}', '{end_str}',\n")

print(f"Saved video times to {output_file}")

# ==== PART 2: Plot the video coverage timeline ====

# Read the saved file to extract timestamps.
with open(output_file, "r") as f:
    content = f.read()

# Use regex to extract all quoted timestamps.
timestamps = re.findall(r"'([^']+)'", content)

# Check that we have an even number (each segment has a start and end).
if len(timestamps) % 2 != 0:
    print("Warning: Expected an even number of timestamps, found", len(timestamps))

# Build a list of segments: each segment is a tuple (start_datetime, end_datetime)
segments = []
for i in range(0, len(timestamps), 2):
    start_str = timestamps[i]
    end_str = timestamps[i+1]
    try:
        start_dt = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S.%f")
        end_dt = datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S.%f")
        segments.append((start_dt, end_dt))
    except Exception as e:
        print("Error parsing:", start_str, end_str, e)
        continue

# Group segments by day (using the date part of the start time)
day_segments = {}
for start_dt, end_dt in segments:
    day = start_dt.date()  # assumes each segment is within one day
    start_sec = start_dt.hour * 3600 + start_dt.minute * 60 + start_dt.second + start_dt.microsecond / 1e6
    duration = (end_dt - start_dt).total_seconds()
    if day not in day_segments:
        day_segments[day] = []
    day_segments[day].append((start_sec, duration))

# Sort days
sorted_days = sorted(day_segments.keys())

# Set up the plot.
fig, ax = plt.subplots(figsize=(10, len(sorted_days) * 0.5 + 2))

# For each day, plot a horizontal bar from 0 to 86400 seconds (full day) in light gray,
# and overlay the video segments in blue.
y_ticks = []
y_labels = []
y = 0
bar_height = 0.8
for day in sorted_days:
    # Full day background (light gray)
    ax.broken_barh([(0, 86400)], (y, bar_height), facecolors='lightgray')
    # Video segments (blue)
    ax.broken_barh(day_segments[day], (y, bar_height), facecolors='blue')
    y_ticks.append(y + bar_height/2)
    y_labels.append(day.strftime("%b %d"))
    y += 1

ax.set_ylim(0, y)
ax.set_xlim(0, 86400)
ax.set_xlabel("Seconds from midnight")
ax.set_ylabel("Day")
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels)
ax.set_title("Video Coverage Timeline for October")
plt.tight_layout()

# Instead of showing the plot, save it to a PNG file.
output_png = "2022_video_coverage.png"
plt.savefig(output_png, dpi=300)
print(f"Saved coverage plot to {output_png}")