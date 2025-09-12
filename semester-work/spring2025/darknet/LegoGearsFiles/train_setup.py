# http://maltserver.cise.ufl.edu:6875/books/betos-book/page/legogears-confusion-matrix-yolov4-w-darknet
import os

legogears_dir = "LegoGears_v2"
dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), legogears_dir)

# Set your dataset subdirectories
dataset_dirs = [
    'set_01',
    'set_02_empty',
    'set_03'
]
dataset_dirs = [os.path.join(dir_path, d) for d in dataset_dirs]

train_file = os.path.join(dir_path, 'LegoGears_train.txt')
valid_file = os.path.join(dir_path, 'LegoGears_valid.txt')
data_file = os.path.join(dir_path, 'LegoGears.data')

# Collect all image file paths from the directories
images = []
for dataset_dir in dataset_dirs:
    for f in os.listdir(dataset_dir):
        if f.lower().endswith('.jpg'):
            images.append(os.path.join(dataset_dir, f))

# Optionally sort the list for consistency
images.sort()

# Write the full list to both train and validation files
with open(train_file, 'w') as tf:
    tf.write('\n'.join(images) + '\n')

with open(valid_file, 'w') as vf:
    vf.write('\n'.join(images) + '\n')

with open(data_file, 'w') as df:
    df.write(f"classes = 5\ntrain = {train_file}\nvalid = {valid_file}\nnames = {os.path.join(dir_path, 'LegoGears.names')}\nbackup = {dir_path}")

print(f"Train file: {train_file}")
print(f"Validation file: {valid_file}")
print(f"Data file: {data_file}")
