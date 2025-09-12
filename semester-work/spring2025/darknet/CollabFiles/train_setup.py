# http://maltserver.cise.ufl.edu:6875/books/betos-book/page/legogears-confusion-matrix-yolov4-w-darknet
import os
import random

# Set your dataset paths
workspace_dir = '/workspace/unzips'
dataset_dirs = []
for d in os.listdir(workspace_dir):
    if not d.endswith('_unzipped'): continue
    if os.path.exists(os.path.join(workspace_dir, d, 'obj_train_data', 'images')):
        dataset_dirs.append(os.path.join(workspace_dir, d, 'obj_train_data', 'images'))
    elif os.path.exists(os.path.join(workspace_dir, d, 'obj_train_data')):
        dataset_dirs.append(os.path.join(workspace_dir, d, 'obj_train_data'))

train_file = '/workspace/unzips/cars_train.txt'
valid_file = '/workspace/unzips/cars_valid.txt'

# Get list of all images from all directories
images = []
for dataset_dir in dataset_dirs:
    for f in os.listdir(dataset_dir):
        if f.endswith('.PNG'):
            images.append(os.path.join(dataset_dir, f))

# Shuffle the images
random.shuffle(images)

# Split into training and validation (80% train, 20% valid)
split_index = int(0.8 * len(images))
train_images = images[:split_index]
valid_images = images[split_index:]

# Write to files
with open(train_file, 'w') as tf:
    tf.write('\n'.join(train_images) + '\n')

with open(valid_file, 'w') as vf:
    vf.write('\n'.join(valid_images) + '\n')

print(f"Train file: {train_file}, Validation file: {valid_file}")