# http://maltserver.cise.ufl.edu:6875/books/betos-book/page/legogears-confusion-matrix-yolov4-w-darknet
import os
import random

# Set your dataset paths
workspace_dir = '/workspace'
dataset_dirs = [os.path.join(workspace_dir, d) for d in os.listdir(workspace_dir) if d.endswith('_unzipped')]


train_file = '/workspace/cars_train.txt'
valid_file = '/workspace/cars_valid.txt'

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