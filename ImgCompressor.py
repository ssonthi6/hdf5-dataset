from random import shuffle
import glob
import numpy as np
import h5py
import cv2

shuffle_data = True
hdf5_path = '/Users/sohamsonthi/Downloads/NN/Data/dataset3.hdf5'  # Where the hdf5 file is saved
train_path = '/Users/sohamsonthi/Downloads/NN/set2/*.jpg' # Where the images are saved
addr = glob.glob(train_path)
labels = [0 if 'cube' in addr else 1 for addr in addr]  # tags as 0 if "cube" is present in the name

if shuffle_data:                    # Rearranges the order of images when transferring to dataset
    c = list(zip(addr, labels))
    shuffle(c)
    addrs, labels = zip(*c)

#   test, train and validation sets broken up into 0.6, 0.2, 0.2
train_addrs = addrs[0:int(0.6*len(addrs))]
train_labels = labels[0:int(0.6 * len(labels))]
val_addrs = addrs[int(0.6 * len(addrs)):int(0.8 * len(addrs))]
val_labels = labels[int(0.6 * len(addrs)):int(0.8 * len(addrs))]
test_addrs = addrs[int(0.8 * len(addrs)):]
test_labels = labels[int(0.8 * len(labels)):]
print("Created train, val, test sets")

# Makes the shape of each the matrices for image conversion (resizes image to 224x224)
train_shape = (len(train_addrs), 224, 224, 3)
val_shape = (len(val_addrs), 224, 224, 3)
test_shape = (len(test_addrs), 224, 224, 3)

# Creates an hdf5 file and creates labels/features
hdf5_file = h5py.File(hdf5_path, mode='w')
hdf5_file.create_dataset("train_img", train_shape, np.int8)
hdf5_file.create_dataset("val_img", val_shape, np.int8)
hdf5_file.create_dataset("test_img", test_shape, np.int8)
hdf5_file.create_dataset("train_mean", train_shape[1:], np.float32)
hdf5_file.create_dataset("train_labels", (len(train_addrs),), np.int8)
hdf5_file["train_labels"][...] = train_labels
hdf5_file.create_dataset("val_labels", (len(val_addrs),), np.int8)
hdf5_file["val_labels"][...] = val_labels
hdf5_file.create_dataset("test_labels", (len(test_addrs),), np.int8)
hdf5_file["test_labels"][...] = test_labels

mean = np.zeros(train_shape[1:], np.float32)

# loop over train set
for i in range(len(train_addrs)):
    if i % 1000 == 0 and i > 1: # Prints how many pictures have been printed (counts every 1000 pictures)
        print('Train data: {}/{}'.format(i, len(train_addrs)))
    addr = train_addrs[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    hdf5_file["train_img"][i, ...] = img[None]
    mean += img / float(len(train_labels))

# loop over validation set
for i in range(len(val_addrs)):

    # print how many images are saved every 1000 images
    if i % 1000 == 0 and i > 1:
        print('Validation data: {}/{}'.format(i, len(val_addrs)))

    addr = val_addrs[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    hdf5_file["val_img"][i, ...] = img[None]

# loop over test set
for i in range(len(test_addrs)):
    # print how many images are saved every 1000 images
    if i % 1000 == 0 and i > 1:
        print('Test data: {}/{}'.format(i, len(test_addrs)))

    addr = test_addrs[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    hdf5_file["test_img"][i, ...] = img[None]

# save the mean and close the hdf5 file
hdf5_file["train_mean"][...] = mean
hdf5_file.close()

print("Completed creating hdf5")

