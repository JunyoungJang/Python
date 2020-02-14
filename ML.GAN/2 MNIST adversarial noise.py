# https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/12_Adversarial_Noise_MNIST.ipynb
# http://wiseodd.github.io/techblog/2016/09/17/gan-tensorflow/


import time
from datetime import timedelta

import functions_from_Hvass as H
import numpy as np
import tensorflow as tf

num_channels = 1
noise_limit = 0.35
noise_l2_weight = 0.02
train_batch_size = 64


# load MNIST data ######################################################################################################
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("data/MNIST/", one_hot=True) # direct download
# data = input_data.read_data_sets("/Users/sungchul/Dropbox/Data/MNIST/", one_hot=True) # local upload

# One-Hot Encoding - data.train.labels, data.test.labels, data.validation.labels
print(data.test.labels[0:5, :])

# Not one-Hot Encoding - data.train.classes, data.test.classes, data.validation.classes
data.train.classes = np.argmax(data.train.labels, axis=1)
data.test.classes = np.argmax(data.test.labels, axis=1)
data.validation.classes = np.argmax(data.validation.labels, axis=1)
print(data.test.classes[0:5])

print("Training set size:\t\t{}".format(len(data.train.images)))
print("Test set size:\t\t\t{}".format(len(data.test.labels)))
print("Validation set size:\t{}".format(len(data.validation.classes)))

# Data dimensions
img_size = 28  # We know that MNIST images are 28 pixels in each dimension.
img_size_flat = img_size * img_size  # Images are stored in one-dimensional arrays of this length.
img_shape = (img_size, img_size)  # Tuple with height and width of images used to reshape arrays.
num_classes = 10  # Number of classes, one class for each of 10 digits.

# Plot a few images to see if data is correct
images = data.test.images[0:9]  # Get the first images from the test-set.
classes_true = data.test.classes[0:9]  # Get the true classes for those images.
H.plot_images(images=images, cls_true=classes_true)
# load MNIST data ######################################################################################################


# functions for the easy model construction
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))

def weight_variable_noise(shape):
    return tf.Variable(0.1*tf.truncated_normal(shape=shape, stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def conv2d(x, W):
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # This is 2x2 max-pooling, which means that we
    # consider 2x2 windows and select the largest value
    # in each window. Then we move 2 pixels to the next window.
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# If we add the noise-variable to the collection  tf.GraphKeys.VARIABLES
# then it will also get initialized with all the other variables in the TensorFlow graph,
# but it will not get optimized. This is a bit confusing.
ADVERSARY_VARIABLES = 'adversary_variables'
collections = [tf.GraphKeys.GLOBAL_VARIABLES, ADVERSARY_VARIABLES]

# Placeholder variables
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)

# Model - input
# input shape (-1,28*28)
# output shape (-1,28,28,1). Here 1 means there is only one channel!
x_image = tf.reshape(x, shape=(-1, 28, 28, 1))
x_noise = tf.Variable(tf.zeros([img_size, img_size, num_channels]), name='x_noise', trainable=False, collections=collections)
x_noise_clip = tf.assign(x_noise, tf.clip_by_value(x_noise, -noise_limit, noise_limit)) # will be executed after optimization-step
x_noisy_image = x_image + x_noise
x_noisy_image = tf.clip_by_value(x_noisy_image, 0.0, 1.0)

# Model - convolution layer 1
# 32 filters of shape (5,5,1)
# input shape (-1,28,28,1)
# output shape (-1,14,14,32)
W_conv1 = weight_variable(shape=(5, 5, 1, 32))
b_conv1 = bias_variable(shape=(1, 32))
h_conv1 = tf.nn.relu(conv2d(x_noisy_image, W_conv1) + b_conv1)
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
h_pool2_flat = tf.reshape(h_pool2, shape=(-1, 7 * 7 * 64))

# Model - fully connected layer 1
# input shape (-1,7*7*64)
# output shape (-1,1024)
W_fc1 = weight_variable(shape=(7 * 7 * 64, 1024))
b_fc1 = bias_variable(shape=(1, 1024))
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Model - fully connected layer 2
# input shape (-1,1024)
# output shape (-1,10)
W_fc2 = weight_variable(shape=(1024, 10))
b_fc2 = bias_variable(shape=(1, 10))
score = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

print([var.name for var in tf.trainable_variables()])

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=score))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

# Optimizer for Adversarial Noise
adversary_variables = tf.get_collection(ADVERSARY_VARIABLES)
print([var.name for var in adversary_variables])
l2_loss_noise = noise_l2_weight * tf.nn.l2_loss(x_noise)
loss_adversary = loss + l2_loss_noise
optimizer_adversary = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(loss_adversary, var_list=adversary_variables)

# Performance measures
prediction = tf.argmax(score, 1)
true_class = tf.argmax(y, 1)
correct_prediction = tf.equal(prediction, true_class)  # boolean for correct prediction
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Create TensorFlow session
with tf.Session() as session:
    tf.global_variables_initializer().run()

    def init_noise():
        session.run(tf.variables_initializer([x_noise]))

    # init_noise()

    def optimize(num_iterations, adversary_target_cls=None):
        start_time = time.time()

        for i in range(num_iterations):
            x_batch, y_true_batch = data.train.next_batch(train_batch_size)
            if adversary_target_cls is not None:
                y_true_batch = np.zeros_like(y_true_batch)
                y_true_batch[:, adversary_target_cls] = 1.0
            feed_dict_train = {x: x_batch,
                               y: y_true_batch,
                               keep_prob: 0.5}

            if adversary_target_cls is None:
                session.run(optimizer, feed_dict=feed_dict_train)
            else:
                session.run(optimizer_adversary, feed_dict=feed_dict_train)
                # Clip / limit the adversarial noise.
                # It cannot be executed in the same session.run() as the optimizer, because
                # it may run in parallel so the execution order is not guaranteed.
                # We need the clip to run after the optimizer.
                session.run(x_noise_clip)

            # Print status every 100 iterations.
            if (i % 100 == 0) or (i == num_iterations - 1):
                acc = session.run(accuracy, feed_dict=feed_dict_train)
                msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
                print(msg.format(i, acc))

        end_time = time.time()
        time_dif = end_time - start_time
        print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

    optimize(num_iterations=1000)
    print(session.run(accuracy, {x:data.test.images, y:data.test.labels, keep_prob:1.0}))
    # init_noise()
    optimize(num_iterations=1000, adversary_target_cls=3)
    print(session.run(accuracy, {x: data.test.images, y: data.test.labels, keep_prob: 1.0}))