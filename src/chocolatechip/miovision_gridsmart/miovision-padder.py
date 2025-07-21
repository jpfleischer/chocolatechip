import os
import subprocess
import argparse
import time
from pathlib import Path
from yaspin import yaspin
from yaspin.spinners import Spinners

def process_videos(source_dir, destination_dir):
    """
    Processes all .mp4 files in the source directory, resizes and pads them using FFmpeg,
    and saves them to the destination directory using CPU-based encoding.
    """
    # Ensure destination directory exists
    os.makedirs(destination_dir, exist_ok=True)

    # List all .mp4 files in the source directory
    source_files = [f for f in Path(source_dir).glob("*.mp4")]
    total_files = len(source_files)
    if total_files == 0:
        print("No .mp4 files found in the source directory.")
        return

    print(f"Found {total_files} video files to process.\n")

    # Process each video file
    for i, source_file in enumerate(source_files, start=1):
        output_file = Path(destination_dir) / source_file.name
        print(f"Processing: {source_file.name} ({i}/{total_files})")

        # Record start time
        start_time = time.time()

        # Start spinner using yaspin
        with yaspin(Spinners.dots, text=f"Processing {source_file.name} ({i}/{total_files})") as spinner:
            try:
                # FFmpeg command using CPU-based libx264 and multiple threads
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-i", str(source_file),  # Input file
                    "-vf", "scale=960:960,pad=1280:960:160:0:black",  # CPU-based scaling and padding
                    "-c:v", "libx264",  # Use CPU-based libx264 encoder
                    "-preset", "fast",  # Fast encoding preset
                    "-threads", "64",   # Use multiple CPU threads
                    str(output_file)    # Output file
                ]

                # Run the FFmpeg command without printing output
                subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                # Calculate elapsed time
                elapsed_time = time.time() - start_time
                spinner.text = f"Finished: {source_file.name} ({i}/{total_files}) in {elapsed_time:.2f} seconds"
                spinner.ok("✅")
            except subprocess.CalledProcessError:
                elapsed_time = time.time() - start_time
                spinner.text = f"Error processing: {source_file.name} ({i}/{total_files}) after {elapsed_time:.2f} seconds"
                spinner.fail("❌")
            except Exception as e:
                elapsed_time = time.time() - start_time
                spinner.text = f"Unexpected error: {e} ({i}/{total_files}) after {elapsed_time:.2f} seconds"
                spinner.fail("❌")

    print("\nAll files have been processed.")

def main():
    parser = argparse.ArgumentParser(description="Batch process .mp4 files using FFmpeg on CPU.")
    parser.add_argument("--source", required=True, help="Source directory containing .mp4 files.")
    parser.add_argument("--destination", required=True, help="Destination directory to save processed videos.")
    args = parser.parse_args()

    # Validate source directory
    if not os.path.isdir(args.source):
        print("Error: Source directory does not exist.")
        return

    # Validate destination directory
    if not os.path.exists(args.destination):
        os.makedirs(args.destination, exist_ok=True)

    # Start processing
    process_videos(args.source, args.destination)

if __name__ == "__main__":
    main()
