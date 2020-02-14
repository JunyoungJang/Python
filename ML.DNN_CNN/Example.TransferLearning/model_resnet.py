from __future__ import division, print_function, absolute_import

from skimage import color, io
from scipy.misc import imresize
import numpy as np
import os
from glob import glob

import tflearn
from tflearn.data_utils import to_categorical
from tflearn.layers.core import input_data
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

### Import picture files
train_path = 'train\\'
train_cat_path = os.path.join(train_path, 'cat\\*.jpg')
train_dog_path = os.path.join(train_path, 'dog\\*.jpg')
train_cat_files = sorted(glob(train_cat_path))
train_dog_files = sorted(glob(train_dog_path))
train_numfiles = len(train_cat_files) + len(train_dog_files)
print(' - Number of train set : ', train_numfiles)
size_image = 64
trainX = np.zeros((train_numfiles, size_image, size_image, 3), dtype='float64')
trainY = np.zeros(train_numfiles)
count = 0
for f in train_cat_files:
    try:
        img = io.imread(f)
        new_img = imresize(img, (size_image, size_image, 3))
        trainX[count] = np.array(new_img)
        trainY[count] = 0
        count += 1
    except:
        continue

for f in train_dog_files:
    try:
        img = io.imread(f)
        new_img = imresize(img, (size_image, size_image, 3))
        trainX[count] = np.array(new_img)
        trainY[count] = 1
        count += 1
    except:
        continue


test_path = 'validation\\'
test_cat_path = os.path.join(test_path, 'cat\\*.jpg')
test_dog_path = os.path.join(test_path, 'dog\\*.jpg')
test_cat_files = sorted(glob(test_cat_path))
test_dog_files = sorted(glob(test_dog_path))
test_numfiles = len(test_cat_files) + len(test_dog_files)
print(' - Number of test set : ', test_numfiles)
testX = np.zeros((test_numfiles, size_image, size_image, 3), dtype='float64')
testY = np.zeros(test_numfiles)
count = 0
for f in test_cat_files:
    try:
        img = io.imread(f)
        new_img = imresize(img, (size_image, size_image, 3))
        testX[count] = np.array(new_img)
        testY[count] = 0
        count += 1
    except:
        continue

for f in test_dog_files:
    try:
        img = io.imread(f)
        new_img = imresize(img, (size_image, size_image, 3))
        testX[count] = np.array(new_img)
        testY[count] = 1
        count += 1
    except:
        continue

trainY = to_categorical(trainY, 2)
testY = to_categorical(testY, 2)


# Image transformations
n = 5
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_crop([64, 64], padding=4)


# Define network
net = input_data(shape=[None, 64, 64, 3], data_preprocessing=img_prep, data_augmentation=img_aug)
net = tflearn.conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001)
net = tflearn.residual_block(net, n, 16)
net = tflearn.residual_block(net, 1, 32, downsample=True)
net = tflearn.residual_block(net, n-1, 32)
net = tflearn.residual_block(net, 1, 64, downsample=True)
net = tflearn.residual_block(net, n-1, 64)
net = tflearn.batch_normalization(net)
net = tflearn.activation(net, 'relu')
net = tflearn.global_avg_pool(net)
# Regression
net = tflearn.fully_connected(net, 2, activation='softmax')
mom = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
net = tflearn.regression(net, optimizer=mom,
                         loss='categorical_crossentropy')
model = tflearn.DNN(net, checkpoint_path='jun_rnn_cat_dog.tflearn',
                    max_checkpoints=1, tensorboard_verbose=3, tensorboard_dir='tmp/tflearn_logs/')
model.fit(trainX, trainY, n_epoch=100, validation_set=(testX, testY), shuffle=True,
          show_metric=True, batch_size=64, snapshot_step=200,
          snapshot_epoch=False, run_id='jun_rnn_cat_dog')
model.save('jun_rnn_cat_dog_final.tflearn')
# rnn typo -> res