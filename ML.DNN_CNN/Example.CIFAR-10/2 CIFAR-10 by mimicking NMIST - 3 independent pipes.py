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

LEARNING_RATE = 0.00025
DECAY_RATE = 0.9
EPOCH_TRAIN = 300
BATCH_SIZE = 128
TEST_SIZE = 256

X_train,Y_train, X_test, Y_test = load_CIFAR10('/Users/sungchul/Dropbox/Data/CIFAR-10/') # a magic function we provide
print( type(X_train), X_train.shape )
print( type(Y_train), Y_train.shape )
print( type(X_test), X_test.shape )
print( type(Y_test), Y_test.shape )
print( Y_test[0:20] )
X_train0 = X_train[:,:,:,0].reshape(-1,32,32,1)
X_train1 = X_train[:,:,:,1].reshape(-1,32,32,1)
X_train2 = X_train[:,:,:,2].reshape(-1,32,32,1)
X_test0 = X_test[:,:,:,0].reshape(-1,32,32,1)
X_test1 = X_test[:,:,:,1].reshape(-1,32,32,1)
X_test2 = X_test[:,:,:,2].reshape(-1,32,32,1)

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

x0 = tf.placeholder(tf.float32)
x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

keep_prob = tf.placeholder(tf.float32)

def from_input_to_output(input, keep_prob, W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2):
    # 32 filters of shape (5,5,1)
    # input shape (-1,32, 32, 1)
    # output shape (-1,16,16,32)
    h_conv1 = tf.nn.relu(conv2d(input, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # 64 filters of shape (5,5,32)
    # input shape (-1,16,16,32)
    # output shape (-1,8,8,64)
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # input shape (-1,8,8,64)
    # output shape (-1,8*8*64)
    h_pool2_flat = tf.reshape(h_pool2, shape=(-1, 8*8*64))

    # input shape (-1,8*8*64)
    # output shape (-1,1024)
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # input shape (-1,1024)
    # output shape (-1,10)
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv

W_conv10 = weight_variable(shape=(5, 5, 1, 32))
b_conv10 = bias_variable(shape=(1, 32))
W_conv20 = weight_variable(shape=(5, 5, 32, 64))
b_conv20 = bias_variable(shape=(1, 64))
W_fc10 = weight_variable(shape=(8 * 8 * 64, 1024))
b_fc10 = bias_variable(shape=(1, 1024))
W_fc20 = weight_variable(shape=(1024, 10))
b_fc20 = bias_variable(shape=(1, 10))

W_conv11 = weight_variable(shape=(5, 5, 1, 32))
b_conv11 = bias_variable(shape=(1, 32))
W_conv21 = weight_variable(shape=(5, 5, 32, 64))
b_conv21 = bias_variable(shape=(1, 64))
W_fc11 = weight_variable(shape=(8 * 8 * 64, 1024))
b_fc11 = bias_variable(shape=(1, 1024))
W_fc21 = weight_variable(shape=(1024, 10))
b_fc21 = bias_variable(shape=(1, 10))

W_conv12 = weight_variable(shape=(5, 5, 1, 32))
b_conv12 = bias_variable(shape=(1, 32))
W_conv22 = weight_variable(shape=(5, 5, 32, 64))
b_conv22 = bias_variable(shape=(1, 64))
W_fc12 = weight_variable(shape=(8 * 8 * 64, 1024))
b_fc12 = bias_variable(shape=(1, 1024))
W_fc22 = weight_variable(shape=(1024, 10))
b_fc22 = bias_variable(shape=(1, 10))

y_conv0 = from_input_to_output(x0, keep_prob, W_conv10, b_conv10, W_conv20, b_conv20, W_fc10, b_fc10, W_fc20, b_fc20)
y_conv1 = from_input_to_output(x1, keep_prob, W_conv11, b_conv11, W_conv21, b_conv21, W_fc11, b_fc11, W_fc21, b_fc21)
y_conv2 = from_input_to_output(x2, keep_prob, W_conv12, b_conv12, W_conv22, b_conv22, W_fc12, b_fc12, W_fc22, b_fc22)

W_final0 = weight_variable(shape=(10, 10))
W_final1 = weight_variable(shape=(10, 10))
W_final2 = weight_variable(shape=(10, 10))
b_final = bias_variable(shape=(1, 10))
y_conv = tf.matmul(y_conv0,W_final0) + tf.matmul(y_conv1,W_final1) + tf.matmul(y_conv2,W_final2) + b_final

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
            sess.run(train, feed_dict={x0: X_train0[start:end,:,:,:], x1: X_train1[start:end,:,:,:], x2: X_train2[start:end,:,:,:], y: Y_train[start:end,:], keep_prob: 0.5})
            #_, cost_now = sess.run([train, cost], feed_dict={x0: X_train0[start:end,:,:,:], x1: X_train1[start:end,:,:,:], x2: X_train2[start:end,:,:,:], y: Y_train[start:end,:], keep_prob: 0.5})
            #print cost_now

            # test_indices = np.arange(len(X_test)) # Get A Test Batch
            # np.random.shuffle(test_indices)
            # test_indices = test_indices[0:TEST_SIZE]
            # print(start, cost_now, np.mean(np.argmax(Y_test[test_indices], axis=1) ==
            #                  sess.run(prediction, feed_dict={x0: X_test0[test_indices], x1: X_test1[test_indices], x2: X_test2[test_indices], keep_prob: 1.0})))

        accuracy_now = accuracy.eval(feed_dict={x0: X_test0, x1: X_test1, x2: X_test2, y: Y_test, keep_prob: 1.0})
        print("EPOCH %4d   test accuracy %g" % (i, accuracy_now))
        if (accuracy_previous > accuracy_now) or (accuracy_now - accuracy_previous  < 0.02):
            LEARNING_RATE = LEARNING_RATE * 0.99
        accuracy_previous = accuracy_now

