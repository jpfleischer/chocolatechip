# http://maltserver.cise.ufl.edu:6875/books/betos-book/page/legogears-confusion-matrix-yolov4-w-darknet
import os
import random

legogears_dir = "LegoGears_v2"
dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), legogears_dir)

# Set your dataset paths
dataset_dirs = [
    'set_01',
    'set_02_empty',
    'set_03'
]

dataset_dirs = [os.path.join(dir_path, dataset_dir) for dataset_dir in dataset_dirs]

train_file = os.path.join(dir_path, 'LegoGears_train.txt')
valid_file = os.path.join(dir_path, 'LegoGears_valid.txt')
data_file = os.path.join(dir_path, 'LegoGears.data')

# Get list of all images from all directories
images = []
for dataset_dir in dataset_dirs:
    for f in os.listdir(dataset_dir):
        if f.endswith('.jpg'):
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

with open(data_file, 'w') as df:
    df.write(f"classes = 5\ntrain = {train_file}\nvalid = {valid_file}\nnames = {os.path.join(dir_path, 'LegoGears.names')}\nbackup = {dir_path}")

print(f"Train file: {train_file}\nValidation file: {valid_file}\nData file {data_file}")
