import time
import os
import sys
import shutil
from datetime import datetime

def get_new_filename(ofile):
    # Mapping of intersection identifiers to camera IDs
    intersec_dict = {
        '61Av_0': 31, '61Av_1': 32,
        '66Av_0': 25, '66Av_1': 26,
        '68Av_0': 24, 'SR7_0': 21,
        'SR7_1': 22, 'Univ_0': 28,
        'Univ_1': 29
    }

    # Month abbreviation to number mapping
    month_dict = {
        'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
        'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
        'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
    }

    filename = os.path.basename(ofile)
    filename_no_ext, ext = os.path.splitext(filename)
    parts = filename_no_ext.split('_')

    # Try parsing as a gridsmart file
    try:
        date_str = parts[0]  # Expected format: 'YYYY-MM-DD'
        datetime.strptime(date_str, '%Y-%m-%d')  # Validate date format

        time_str = parts[1]  # Expected format: 'HH-MM-SS-rtsp'
        time_parts = time_str.split('-')
        hour, minute, second = time_parts[:3]

        site_name = parts[2]  # e.g., 'Stirling-61Av'
        camera_suffix = parts[3]  # e.g., '0'

        # Construct intersection key
        intersec_key = site_name.split('-')[-1] + '_' + camera_suffix  # e.g., '61Av_0'
        camera_id = str(intersec_dict[intersec_key])

        # Build new filename
        date_time = f"{date_str}_{hour}-{minute}-{second}"
        new_filename = f"{camera_id}_{date_time}.000.mp4"
        return new_filename
    except (IndexError, ValueError, KeyError):
        # Not a gridsmart file; try parsing as a miovision file
        pass

    # Try parsing as a miovision file
    try:
        # Split from the right to handle filenames with varying lengths
        parts = filename_no_ext.rsplit('_', 6)
        uuid_and_month, day, year, hour, minute, second, millisecond = parts

        # Determine camera ID based on the filename prefix
        camera_id = '34' if filename.startswith('2e747114') else '35'

        # Extract month abbreviation
        month_str = uuid_and_month[-3:]
        month = month_dict[month_str]

        date_str = f"{year}-{month}-{day}"
        time_str = f"{hour}-{minute}-{second}"

        new_filename = f"{camera_id}_{date_str}_{time_str}.{millisecond.zfill(3)}.mp4"
        return new_filename
    except (IndexError, KeyError):
        # Unable to parse filename
        return None

def process_file(source_path, destination_dir, dryrun):
    filename = os.path.basename(source_path)
    
    # Skip if it's not an .mp4 file
    if not filename.lower().endswith('.mp4'):
        print(f"Skipping '{filename}': Not an mp4 file.")
        return

    new_filename = get_new_filename(source_path)
    if new_filename:
        destination_path = os.path.join(destination_dir, new_filename)
        if dryrun:
            print(f"[DRY RUN] Would copy '{source_path}' to '{destination_path}'")
        else:
            print(f"Copying '{source_path}' to '{destination_path}'")
            try:
                # time.sleep(2)
                shutil.copy2(source_path, destination_path)
                # Update the modification time to the current time
                # this is crucial or else nifi wont see it.
                os.utime(destination_path, None)
            except Exception as e:
                print(f"Failed to copy '{source_path}' to '{destination_path}': {e}")
    else:
        print(f"Warning: Unable to process file '{filename}'. Skipping.")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Process and rename video files.')
    parser.add_argument('source', help='Source file or directory containing video files.')
    parser.add_argument('destination_dir', help='Destination directory for processed files.')
    parser.add_argument('--dryrun', action='store_true', help="Print the file operations without performing them.")
    args = parser.parse_args()

    source = args.source
    destination_dir = args.destination_dir
    dryrun = args.dryrun

    # Validate destination directory
    if not dryrun and not os.path.isdir(destination_dir):
        print(f"Error: Destination directory '{destination_dir}' does not exist.")
        sys.exit(1)

    # Check if source is a file
    if os.path.isfile(source):
        process_file(source, destination_dir, dryrun)
    # Check if source is a directory
    elif os.path.isdir(source):
        # Process each file in the source directory
        # only one dir deep which is great
        for filename in os.listdir(source):
            source_path = os.path.join(source, filename)
            if os.path.isfile(source_path):
                process_file(source_path, destination_dir, dryrun)
    else:
        print(f"Error: Source '{source}' does not exist.")
        sys.exit(1)

if __name__ == '__main__':
    main()
