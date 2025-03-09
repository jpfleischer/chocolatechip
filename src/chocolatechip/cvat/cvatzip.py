#!/usr/bin/env python3
import os
import shutil
import tempfile
import zipfile

# Configuration
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
SUBSET_FOLDER = "obj_train_data"
TRAIN_LIST_FILENAME = "train.txt"
ZIP_FILENAME = "dataset.zip"
OBJ_DATA_FILENAME = "obj.data"
OBJ_NAMES_FILENAME = "obj.names"
BACKUP_FOLDER = "backup"

# Content for metadata files
OBJ_DATA_CONTENT = """classes = 5
train = train.txt
names = obj.names
backup = backup/
"""
OBJ_NAMES_CONTENT = """motorbike
car
truck
bus
pedestrian
"""

def is_image_file(filename):
    _, ext = os.path.splitext(filename)
    return ext in IMAGE_EXTENSIONS

def cvatzip():
    cwd = os.getcwd()

    # Find image files in the current directory
    image_files = [f for f in os.listdir(cwd) if os.path.isfile(f) and is_image_file(f)]
    if not image_files:
        print("No image files found in the current directory.")
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create the subset folder (for images and annotation files)
        subset_dir = os.path.join(temp_dir, SUBSET_FOLDER)
        os.makedirs(subset_dir, exist_ok=True)

        # Copy image files and corresponding annotation files (if any)
        for img in image_files:
            shutil.copy2(os.path.join(cwd, img), subset_dir)
            base, _ = os.path.splitext(img)
            annotation_file = base + ".txt"
            if os.path.isfile(annotation_file):
                shutil.copy2(os.path.join(cwd, annotation_file), subset_dir)

        # Create train.txt at the root of temp_dir listing relative paths for each image file
        train_list_path = os.path.join(temp_dir, TRAIN_LIST_FILENAME)
        with open(train_list_path, "w") as f:
            for img in image_files:
                f.write(f"{SUBSET_FOLDER}/{img}\n")

        # Create obj.names at the root of temp_dir
        obj_names_path = os.path.join(temp_dir, OBJ_NAMES_FILENAME)
        with open(obj_names_path, "w") as f:
            f.write(OBJ_NAMES_CONTENT)

        # Create obj.data at the root of temp_dir
        obj_data_path = os.path.join(temp_dir, OBJ_DATA_FILENAME)
        with open(obj_data_path, "w") as f:
            f.write(OBJ_DATA_CONTENT)

        # Create an empty backup folder as referenced by obj.data
        backup_dir = os.path.join(temp_dir, BACKUP_FOLDER)
        os.makedirs(backup_dir, exist_ok=True)

        # Create the ZIP archive with the proper structure
        zip_path = os.path.join(cwd, ZIP_FILENAME)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Compute the relative path in the archive
                    rel_path = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arcname=rel_path)

        print(f"Created {ZIP_FILENAME} with the following structure:")
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            for name in zipf.namelist():
                print(name)

if __name__ == "__main__":
    cvatzip()
