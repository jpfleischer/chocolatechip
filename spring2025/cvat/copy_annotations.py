#!/usr/bin/env python3
import os
import shutil
import re

def number_in_range(file_name, min_val=0, max_val=475):
    """
    Extracts the first sequence of digits from file_name and returns True 
    if the integer is between min_val and max_val (inclusive).
    """
    m = re.search(r'(\d+)', file_name)
    if m:
        try:
            number = int(m.group(1))
            return min_val <= number <= max_val
        except ValueError:
            return False
    return False

def main():
    # Determine the directory where this script resides.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Assume the "docker/unzips" folder is relative to the script directory.
    base_dir = os.path.join(script_dir, "docker", "unzips")
    
    # Destination directory (absolute) where annotations will be stored.
    base_dest_dir = "/data/traffic_training_cvat_unzipped"
    # Subdirectory "annotations" inside the destination.
    dest_dir = os.path.join(base_dest_dir, "annotations")
    
    # Create destination directory (and annotations subdirectory) if it doesn't exist.
    os.makedirs(dest_dir, exist_ok=True)
    
    # Define task-specific numeric ranges.
    task_ranges = {
        "76": (0, 475),
        "81": (0, 636)
    }
    
    # Loop through all entries in the base_dir.
    for entry in os.listdir(base_dir):
        # Determine if this folder is one of the tasks with a numeric range
        task_range = None
        for task, (min_val, max_val) in task_ranges.items():
            if task in entry:
                task_range = (min_val, max_val)
                break

        images_dir = os.path.join(base_dir, entry, "obj_train_data", "images")
        if os.path.isdir(images_dir):
            print(f"Processing folder: {images_dir}")
            for file_name in os.listdir(images_dir):
                # Only process .jpg, .png, and .txt files.
                if not (file_name.lower().endswith(".jpg") or 
                        file_name.lower().endswith(".png") or 
                        file_name.lower().endswith(".txt")):
                    continue
                # Enforce range check if applicable.
                if task_range and not number_in_range(file_name, *task_range):
                    continue
                src_file = os.path.join(images_dir, file_name)
                dest_file_name = f"{entry}_{file_name}"
                dest_file = os.path.join(dest_dir, dest_file_name)
                try:
                    shutil.copy2(src_file, dest_file)
                    print(f"Copied {src_file} -> {dest_file}")
                except Exception as e:
                    print(f"Error copying {src_file}: {e}")
        else:
            print(f"Images folder not found: {images_dir}. Checking alternative files in {entry} ...")
            alternative_dir = os.path.join(base_dir, entry)
            for root, _, files in os.walk(alternative_dir):
                for file_name in files:
                    # Skip train.txt as in the original script.
                    if file_name.lower() == "train.txt":
                        continue
                    # Process .jpg, .png, and .txt files.
                    if not (file_name.lower().endswith(".jpg") or 
                            file_name.lower().endswith(".png") or 
                            file_name.lower().endswith(".txt")):
                        continue
                    if task_range and not number_in_range(file_name, *task_range):
                        continue
                    src_file = os.path.join(root, file_name)
                    dest_file_name = f"{entry}_{file_name}"
                    dest_file = os.path.join(dest_dir, dest_file_name)
                    try:
                        shutil.copy2(src_file, dest_file)
                        print(f"Copied {src_file} -> {dest_file}")
                    except Exception as e:
                        print(f"Error copying {src_file}: {e}")
    
    # Also copy the cars.names file.
    cars_names_src = os.path.join(script_dir, "..", "darknet", "CollabFiles", "cars.names")
    cars_names_dest = os.path.join(dest_dir, "cars.names")
    
    if os.path.isfile(cars_names_src):
        try:
            shutil.copy2(cars_names_src, cars_names_dest)
            print(f"Copied {cars_names_src} -> {cars_names_dest}")
        except Exception as e:
            print(f"Error copying {cars_names_src}: {e}")
    else:
        print(f"cars.names not found at {cars_names_src}")

if __name__ == "__main__":
    main()
