import keras
import os, shutil

print("keras.__version__ = ", keras.__version__)
# keras.__version__ =  2.2.2

# The path to the directory where the original
# dataset was uncompressed
original_dataset_dir = '/media/boom/HDD/FanBu/资料/PhD/research/dogs_vs_cats_rearr'

# The directory where we will
# store our smaller dataset
base_dir = '/media/boom/HDD/FanBu/资料/PhD/research/cats_and_dogs_small'
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

# Destination directories for our training, validation and test splits
train_dir = os.path.join(base_dir, 'train')
if not os.path.exists(train_dir):
    os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
if not os.path.exists(validation_dir):
    os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
if not os.path.exists(test_dir):
    os.mkdir(test_dir)

# Destination directory with our training cat pictures
train_cats_dir = os.path.join(train_dir, 'cats')
if not os.path.exists(train_cats_dir):
    os.mkdir(train_cats_dir)

# Destination directory with our training dog pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')
if not os.path.exists(train_dogs_dir):
    os.mkdir(train_dogs_dir)

# Destination directory with our validation cat pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')
if not os.path.exists(validation_cats_dir):
    os.mkdir(validation_cats_dir)

# Destination directory with our validation dog pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
if not os.path.exists(validation_dogs_dir):
    os.mkdir(validation_dogs_dir)

# Destination directory with our test cat pictures
test_cats_dir = os.path.join(test_dir, 'cats')
if not os.path.exists(test_cats_dir):
    os.mkdir(test_cats_dir)

# Destination directory with our test dog pictures
test_dogs_dir = os.path.join(test_dir, 'dogs')
if not os.path.exists(test_dogs_dir):
    os.mkdir(test_dogs_dir)

# Copy first 1000 cat images to train_cats_dir
print("Copy 1000 cat images to train_cats_dir")
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
original_train_dir = os.path.join(original_dataset_dir, 'train_rearr')
original_train_cats_dir = os.path.join(original_train_dir, 'cats')
for fname in fnames:
    src = os.path.join(original_train_cats_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    # print("src = ", src)
    # print("dst = ", dst)
    shutil.copyfile(src, dst)

# Copy next 500 cat images to validation_cats_dir
print("Copy 500 cat images to validation_cats_dir")
fnames = ['cat.{}.jpg'.format(i) for i in range(10000, 10500)]
original_val_dir = os.path.join(original_dataset_dir, 'val_rearr')
original_val_cats_dir = os.path.join(original_val_dir, 'cats')
for fname in fnames:
    src = os.path.join(original_val_cats_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)

# Copy next 500 cat images to test_cats_dir
print("Copy 500 cat images to test_cats_dir")
fnames = ['cat.{}.jpg'.format(i) for i in range(12000, 12500)]
original_test_dir = os.path.join(original_dataset_dir, 'test_rearr')
original_test_cats_dir = os.path.join(original_test_dir, 'cats')
for fname in fnames:
    src = os.path.join(original_test_cats_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)

# Copy first 1000 dog images to train_dogs_dir
print("Copy 1000 dog images to train_dogs_dir")
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
original_train_dogs_dir = os.path.join(original_train_dir, 'dogs')
for fname in fnames:
    src = os.path.join(original_train_dogs_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    # print("src = ", src)
    # print("dst = ", dst)
    shutil.copyfile(src, dst)

# Copy next 500 dog images to validation_dogs_dir
print("Copy 500 dog images to validation_dogs_dir")
fnames = ['dog.{}.jpg'.format(i) for i in range(10000, 10500)]
original_val_dogs_dir = os.path.join(original_val_dir, 'dogs')
for fname in fnames:
    src = os.path.join(original_val_dogs_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)

# Copy next 500 dog images to test_dogs_dir
print("Copy 500 dog images to test_dogs_dir")
fnames = ['dog.{}.jpg'.format(i) for i in range(12000, 12500)]
original_test_dogs_dir = os.path.join(original_test_dir, 'dogs')
for fname in fnames:
    src = os.path.join(original_test_dogs_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)

print('total training cat images:', len(os.listdir(train_cats_dir)))
print('total training dog images:', len(os.listdir(train_dogs_dir)))
print('total validation cat images:', len(os.listdir(validation_cats_dir)))
print('total validation dog images:', len(os.listdir(validation_dogs_dir)))
print('total test cat images:', len(os.listdir(test_cats_dir)))
print('total test dog images:', len(os.listdir(test_dogs_dir)))
