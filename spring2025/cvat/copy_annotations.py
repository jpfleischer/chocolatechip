#!/usr/bin/env python3
import os
import shutil

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
    
    # Loop through all entries in the base_dir
    for entry in os.listdir(base_dir):
        # Check if the directory name ends with "_unzipped"
        if entry.endswith("_unzipped"):
            images_dir = os.path.join(base_dir, entry, "obj_train_data", "images")
            if os.path.isdir(images_dir):
                print(f"Processing folder: {images_dir}")
                for file_name in os.listdir(images_dir):
                    src_file = os.path.join(images_dir, file_name)
                    # Create a unique destination file name by prefixing with the parent folder name
                    dest_file_name = f"{entry}_{file_name}"
                    dest_file = os.path.join(dest_dir, dest_file_name)
                    try:
                        shutil.copy2(src_file, dest_file)
                        print(f"Copied {src_file} -> {dest_file}")
                    except Exception as e:
                        print(f"Error copying {src_file}: {e}")
            else:
                print(f"Images folder not found: {images_dir}")
    
    # Now, also copy the cars.names file from "../darknet/CollabFiles/cars.names" to the annotations folder.
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
