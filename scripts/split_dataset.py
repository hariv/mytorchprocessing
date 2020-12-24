import os
import sys
import math
import random
import shutil

TRAIN_SPLIT_RATIO = 0.9

def create_split_dirs(data_path):
    if (os.path.exists(os.path.join(data_path, 'train'))
        and os.path.exists(os.path.join(data_path, 'val'))):
        print("Already created train and val dirs")
        exit()
    os.makedirs(os.path.join(data_path, 'train'))
    os.makedirs(os.path.join(data_path, 'val'))

def move(data_path, c, split_dir, samples):
    for s in samples:
        shutil.move(os.path.join(data_path, c, s),
                    os.path.join(data_path, split_dir, c))

def split(data_path):
    classes = os.listdir(data_path)
    create_split_dirs(data_path)
    
    for c in classes:
        os.makedirs(os.path.join(data_path, 'train', c))
        os.makedirs(os.path.join(data_path, 'val', c))

        class_samples = os.listdir(os.path.join(data_path, c))
        
        num_samples = len(class_samples)
        num_train_samples = math.floor(num_samples * TRAIN_SPLIT_RATIO)
        
        train_samples = random.sample(class_samples, num_train_samples)
        val_samples = list(set(class_samples) - set(train_samples))

        move(data_path, c, 'train', train_samples)
        move(data_path, c, 'val', val_samples)

        shutil.rmtree(os.path.join(data_path, c))
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python3 split_dataset.py <path_to_data>')
        exit()
    split(sys.argv[1])
