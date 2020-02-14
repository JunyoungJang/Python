# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
    Simple, end-to-end, LeNet-5-like convolutional MNIST model example.
    
    This should achieve a test error of 0.7%. Please keep this model as simple and
    linear as possible, it is meant as a tutorial for simple convolutional models.
    Run with --self_test on the command line to execute a short self-test.
"""

# https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/02_Convolutional_Neural_Network.ipynb

import time
from datetime import timedelta

import functions_from_Hvass as H
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.misc import imread

# Control panel
LEARNING_RATE = 1e-3
DECAY_RATE = 0.9
EPOCH_TRAIN = 100
BATCH_SIZE = 128
TEST_SIZE = 256

# The following chart shows roughly how the data flows in the Convolutional Neural Network that is implemented below.
img1 = imread('network_flowchart.png')
plt.imshow(img1)
plt.show()

# The following chart shows the basic idea of processing an image in the first convolutional layer.
img2 = imread('convolution.png')
plt.imshow(img2)
plt.show()

# Load Data
# The MNIST data-set is about 12 MB and will be downloaded automatically if it is not located in the given path.
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("data/MNIST/", one_hot=True)
# data = input_data.read_data_sets("/Users/sungchul/Dropbox/Data/MNIST/", one_hot=True)

# print data info
# data.train.images
# data.train.labels - One-Hot encoding (see below)
# data.train.classes - Not one-hot encoding (see below)
# data.train.num_examples
# batch_xs, batch_ys = data.train.next_batch(BATCH_SIZE)
print("Training set size:\t\t{}".format(len(data.train.labels)))
# data.test.images
# data.test.labels - One-Hot encoding (see below)
# data.test.classes - Not one-hot encoding (see below)
# data.test.num_examples
# batch_xs, batch_ys = data.test.next_batch(BATCH_SIZE)
print("Test set size:\t\t\t{}".format(len(data.test.labels)))
# data.validation.images
# data.validation.labels - One-Hot encoding (see below)
# data.validation.classes - Not one-hot encoding (see below)
# data.validation.num_examples
# batch_xs, batch_ys = data.validation.next_batch(BATCH_SIZE)
print("Validation set size:\t{}".format(len(data.validation.labels)))

# One-Hot Encoding - data.train.labels, data.test.labels, data.validation.labels
print(data.test.labels[0:5, :])

# Not one-Hot Encoding - data.train.classes, data.test.classes, data.validation.classes
data.train.classes = np.argmax(data.train.labels, axis=1)
data.test.classes = np.argmax(data.test.labels, axis=1)
data.validation.classes = np.argmax(data.validation.labels, axis=1)
print(data.test.classes[0:5])

# Data dimensions
img_size = 28 # We know that MNIST images are 28 pixels in each dimension.
img_size_flat = img_size * img_size # Images are stored in one-dimensional arrays of this length.
img_shape = (img_size, img_size) # Tuple with height and width of images used to reshape arrays.
num_classes = 10 # Number of classes, one class for each of 10 digits.

# Plot a few images to see if data is correct
images = data.test.images[0:9] # Get the first images from the test-set.
classes_true = data.test.classes[0:9] # Get the true classes for those images.
H.plot_images(images=images, cls_true=classes_true)

# functions for the easy model construction
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))

# functions for the easy model construction
def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

# functions for the easy model construction
def conv2d(x, W):
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# functions for the easy model construction
def max_pool_2x2(x):
    # This is 2x2 max-pooling, which means that we
    # consider 2x2 windows and select the largest value
    # in each window. Then we move 2 pixels to the next window.
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Placeholder variables
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)

# Model - input
# input shape (-1,28*28)
# output shape (-1,28,28,1). Here 1 means there is only one channel!
x_image = tf.reshape(x, shape=(-1,28,28,1))

# Model - convolution layer 1
# 32 filters of shape (5,5,1)
# input shape (-1,28,28,1)
# output shape (-1,14,14,32)
W_conv1 = weight_variable(shape=(5, 5, 1, 32))
b_conv1 = bias_variable(shape=(1, 32))
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Model - convolution layer 2
# 64 filters of shape (5,5,32)
# input shape (-1,14,14,32)
# output shape (-1,7,7,64)
W_conv2 = weight_variable(shape=(5, 5, 32, 64))
b_conv2 = bias_variable(shape=(1, 64))
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Model - flattening output of convolution layer 2
# input shape (-1,7,7,64)
# output shape (-1,7*7*64)
h_pool2_flat = tf.reshape(h_pool2, shape=(-1, 7*7*64))

# Model - fully connected layer 1
# input shape (-1,7*7*64)
# output shape (-1,1024)
W_fc1 = weight_variable(shape=(7*7*64, 1024))
b_fc1 = bias_variable(shape=(1, 1024))
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Model - fully connected layer 2
# input shape (-1,1024)
# output shape (-1,10)
W_fc2 = weight_variable(shape=(1024, 10))
b_fc2 = bias_variable(shape=(1, 10))
score = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Cost to be optimized
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=score))

# Optimization method
train = tf.train.RMSPropOptimizer(LEARNING_RATE, DECAY_RATE).minimize(cost)

# Performance measures
prediction = tf.argmax(score,1)
correct_prediction = tf.equal(prediction, tf.argmax(y,1)) # boolean for correct prediction
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Create TensorFlow session
with tf.Session() as sess:

    # Initialize variables
    tf.global_variables_initializer().run()

    # train
    for i in range(EPOCH_TRAIN):
        start_time = time.time() # Start-time used for printing time-usage below.
        training_batch = zip(range(0, len(data.train.labels), BATCH_SIZE), range(BATCH_SIZE, len(data.train.labels), BATCH_SIZE))
        for start, end in training_batch:
            _, cost_now = sess.run([train, cost], feed_dict={x: data.train.images[start:end,:], y: data.train.labels[start:end,:], keep_prob: 0.5})
            #print(cost_now)
        end_time = time.time() # Ending time.
        time_dif = end_time - start_time # Difference between start and end-times.
        test_indices = np.arange(len(data.test.labels)) # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:TEST_SIZE]
        print("==========================================================")
        print("Epoch:", i)
        print("Accuracy: %g" % sess.run(accuracy, feed_dict={x: data.test.images[:TEST_SIZE,:], y: data.test.labels[:TEST_SIZE,:], keep_prob: 1.0}))
        print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))  # Print the time-usage.

    # test
    print("Test accuracy: %g" % sess.run(accuracy, feed_dict={x: data.test.images, y: data.test.labels, keep_prob: 1.0}))

    # visualization - two images chosen
    image1 = data.test.images[0:1]
    H.plot_image(image1, img_shape)
    # image2 = data.test.images[13:14]
    # H.plot_image(image2, img_shape)

    # visualization - first layer weights and images
    weights_conv1 = sess.run(W_conv1)
    bias_conv1 = sess.run(b_conv1)
    print(weights_conv1.shape)
    H.plot_conv_weights(weights=weights_conv1)
    image1_conv1 = sess.run(max_pool_2x2(tf.nn.relu(tf.add(conv2d(tf.reshape(image1,shape=(-1,28,28,1)),weights_conv1),bias_conv1))))
    H.plot_conv_images(values=image1_conv1)

    # visualization - second layer weights
    weights_conv2 = sess.run(W_conv2)
    bias_conv2 = sess.run(b_conv2)
    print(weights_conv2.shape)
    H.plot_conv_weights(weights=weights_conv2)
    image1_conv2 = sess.run(max_pool_2x2(tf.nn.relu(tf.add(conv2d(image1_conv1, weights_conv2), bias_conv2))))
    H.plot_conv_images(values=image1_conv2)





