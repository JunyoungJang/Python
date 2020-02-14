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

import tensorflow as tf
# from cs231n_utilities import load_CIFAR10
import numpy as np
import os
import _pickle as pickle

LEARNING_RATE = 1e-3
DECAY_RATE = 0.9
EPOCH_TRAIN = 300
BATCH_SIZE = 128
TEST_SIZE = 256


def load_CIFAR_batch(filename):
    """
    modified from cs231n_utilities.py due to python version update
    """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1');
        X = datadict['data'];
        Y = datadict['labels'];
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float");
        Y = np.array(Y);
        return X, Y;


def load_CIFAR10(ROOT):
    """
    from cs231n_utilities.py
    """
    xs = [];
    ys = [];
    for b in range(1, 6):
        f = os.path.join(ROOT, "data_batch_%d" % (b,));
        X, Y = load_CIFAR_batch(f);
        xs.append(X);
        ys.append(Y);
    Xtr = np.concatenate(xs);
    Ytr = np.concatenate(ys);
    del X, Y;
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, "test_batch"));
    return Xtr, Ytr, Xte, Yte;

X_train,Y_train, X_test, Y_test = load_CIFAR10('/Users/sungchul/Dropbox/Data/CIFAR-10/') # a magic function we provide
print( type(X_train), X_train.shape )
print( type(Y_train), Y_train.shape )
print( type(X_test), X_test.shape )
print( type(Y_test), Y_test.shape )
print( Y_test[0:20] )

def one_hot_encode(Y, list_of_class_numbers):
    Y_one_hot = np.zeros((len(Y), len(list_of_class_numbers)))
    for i, class_number in enumerate(list_of_class_numbers):
        Y_one_hot[Y==class_number, i] = 1
    return Y_one_hot

Y_train = one_hot_encode(Y_train, list_of_class_numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
Y_test = one_hot_encode(Y_test, list_of_class_numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print( Y_test[0:20, :] )

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

keep_prob = tf.placeholder(tf.float32)

# 32 filters of shape (5,5,3)
# input shape (-1,32, 32, 3)
# output shape (-1,16,16,32)
W_conv1 = weight_variable(shape=(5, 5, 3, 32))
b_conv1 = bias_variable(shape=(1, 32))
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 64 filters of shape (5,5,32)
# input shape (-1,16,16,32)
# output shape (-1,8,8,64)
W_conv2 = weight_variable(shape=(5, 5, 32, 64))
b_conv2 = bias_variable(shape=(1, 64))
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# input shape (-1,8,8,64)
# output shape (-1,8*8*64)
h_pool2_flat = tf.reshape(h_pool2, shape=(-1, 8*8*64))

# input shape (-1,8*8*64)
# output shape (-1,1024)
W_fc1 = weight_variable(shape=(8*8*64, 1024))
b_fc1 = bias_variable(shape=(1, 1024))
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# input shape (-1,1024)
# output shape (-1,10)
W_fc2 = weight_variable(shape=(1024, 10))
b_fc2 = bias_variable(shape=(1, 10))
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
train = tf.train.RMSPropOptimizer(LEARNING_RATE, DECAY_RATE).minimize(cost)
prediction = tf.argmax(y_conv,1)
correct_prediction = tf.equal(prediction, tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Launch the graph in a session
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    accuracy_previous = 0
    for i in range(EPOCH_TRAIN):
        training_batch = zip(range(0, len(X_train), BATCH_SIZE), range(BATCH_SIZE, len(X_train), BATCH_SIZE))
        for start, end in training_batch:
            # print(start, end)
            # print(X_train[start:end, :].shape)
            # print(Y_train[start:end,:].shape)
            # print(X_train[start:start+1, :])
            # _, cost_now = sess.run([train, cost], feed_dict={x: X_train[start:end,:,:,:], y: Y_train[start:end,:], keep_prob: 0.5})
            # print cost_now
            sess.run(train, feed_dict={x: X_train[start:end, :, :, :], y: Y_train[start:end, :], keep_prob: 0.5})

            # test_indices = np.arange(len(X_test)) # Get A Test Batch
            # np.random.shuffle(test_indices)
            # test_indices = test_indices[0:TEST_SIZE]
            # print(i, np.mean(np.argmax(Y_test[test_indices], axis=1) ==
            #                  sess.run(prediction, feed_dict={x: X_test[test_indices], keep_prob: 1.0})))

        accuracy_now = accuracy.eval(feed_dict={x: X_test, y: Y_test, keep_prob: 1.0})
        print("EPOCH %4d   test accuracy %g" % (i, accuracy_now))
        if (accuracy_previous > accuracy_now) or (accuracy_now - accuracy_previous  < 0.02):
            LEARNING_RATE = LEARNING_RATE * 0.99
        accuracy_previous = accuracy_now

