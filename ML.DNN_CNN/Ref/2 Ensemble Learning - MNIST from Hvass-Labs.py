# https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/05_Ensemble_Learning.ipynb

import os
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf

# Control panel
LEARNING_RATE = 1e-4
# We will train 5 neural networks on different training-sets that are selected at random.
# First we combine the original training- and validation-sets into one big set.
train_size = 50000 # the size of the training-set used for each neural network
train_batch_size = 128
num_networks = 5 # Number of neural networks in the ensemble.
num_iterations = 1000#10000 # Number of optimization iterations for each neural network.

# # The following chart shows roughly how the data flows in the Convolutional Neural Network that is implemented below.
# img1 = imread('network_flowchart.png')
# plt.imshow(img1)
# plt.show()
#
# # The following chart shows the basic idea of processing an image in the first convolutional layer.
# img2 = imread('convolution.png')
# plt.imshow(img2)
# plt.show()

# Load Data
# The MNIST data-set is about 12 MB and will be downloaded automatically if it is not located in the given path.
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("data/MNIST/", one_hot=True)
# data = input_data.read_data_sets("/Users/sungchul/Dropbox/Data/MNIST/", one_hot=True)

# print data info
# data.train.images
# data.train.labels - One-Hot encoding (see below)
# data.train.cls - Not one-hot encoding (see below)
# data.train.num_examples
# batch_xs, batch_ys = data.train.next_batch(BATCH_SIZE)
print("Training set size:\t\t{}".format(len(data.train.labels))) # Training set size:		55000
# data.test.images
# data.test.labels - One-Hot encoding (see below)
# data.test.cls - Not one-hot encoding (see below)
# data.test.num_examples
# batch_xs, batch_ys = data.test.next_batch(BATCH_SIZE)
print("Test set size:\t\t\t{}".format(len(data.test.labels))) # Test set size:			10000
# data.validation.images
# data.validation.labels - One-Hot encoding (see below)
# data.validation.cls - Not one-hot encoding (see below)
# data.validation.num_examples
# batch_xs, batch_ys = data.validation.next_batch(BATCH_SIZE)
print("Validation set size:\t{}".format(len(data.validation.labels))) # Validation set size:	5000

# One-Hot Encoding - data.train.labels, data.test.labels, data.validation.labels
print(data.test.labels[0:5, :])

# Not one-Hot Encoding - data.train.cls, data.test.cls, data.validation.cls
data.train.cls = np.argmax(data.train.labels, axis=1)
data.test.cls = np.argmax(data.test.labels, axis=1)
data.validation.cls = np.argmax(data.validation.labels, axis=1)
print(data.test.cls[0:5])

# Combine train and validation set
combined_images = np.concatenate([data.train.images, data.validation.images], axis=0)
combined_labels = np.concatenate([data.train.labels, data.validation.labels], axis=0)
combined_cls = np.argmax(combined_labels, axis=1)
# Check that the shape of the combined arrays is correct.
print(combined_images.shape) # (60000, 784)
print(combined_labels.shape) # (60000, 10)
print(combined_cls.shape) # (60000,)
# Size of the combined data-set.
combined_size = len(combined_images)
print(combined_size)
train_size = train_size
print(train_size)
validation_size = combined_size - train_size
print(validation_size)

# Helper-function for splitting the combined data-set into a random training- and validation-set.
# def random_training_set(combined_size, train_size, combined_images, combined_labels):
def random_training_set():
    # Create a randomized index into the full / combined training-set.
    idx = np.random.permutation(combined_size)

    # Split the random index into training- and validation-sets.
    idx_train = idx[0:train_size]
    idx_validation = idx[train_size:]

    # Select the images and labels for the new training-set.
    x_train = combined_images[idx_train, :]
    y_train = combined_labels[idx_train, :]

    # Select the images and labels for the new validation-set.
    x_validation = combined_images[idx_validation, :]
    y_validation = combined_labels[idx_validation, :]

    # Return the new training- and validation-sets.
    return x_train, y_train, x_validation, y_validation

# Function for selecting a random training-batch of the given size.
def random_batch(x_train, y_train, batch_size=train_batch_size):

    num_images = len(x_train)

    idx = np.random.choice(num_images,
                           size=batch_size,
                           replace=False)

    x_batch = x_train[idx, :]  # Images.
    y_batch = y_train[idx, :]  # Labels.

    return x_batch, y_batch

# Data dimensions
img_size = 28 # We know that MNIST images are 28 pixels in each dimension.
img_size_flat = img_size * img_size # Images are stored in one-dimensional arrays of this length.
img_shape = (img_size, img_size) # Tuple with height and width of images used to reshape arrays.
num_cls = 10 # Number of cls, one class for each of 10 digits.

# # Plot a few images to see if data is correct
# images = data.test.images[0:9] # Get the first images from the test-set.
# cls_true = data.test.cls[0:9] # Get the true cls for those images.
# H.plot_images(images=images, cls_true=cls_true)

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
optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)
# train = tf.train.RMSPropOptimizer(LEARNING_RATE, DECAY_RATE).minimize(cost)

# Performance measures
y_true = tf.argmax(y,1)
y_pred = tf.argmax(score,1)
correct_prediction = tf.equal(y_pred, y_true) # boolean for correct prediction
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Saver
saver = tf.train.Saver(max_to_keep=100)
save_dir = 'checkpoints/'
# Create the directory if it does not exist.
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# This function returns the save-path for the data-file with the given network number.
def get_save_path(net_number):
    return save_dir + 'network' + str(net_number)

# Create TensorFlow session
with tf.Session() as session:

    # Helper-function to perform optimization iterations
    def optimize(num_iterations, x_train, y_train):
        start_time = time.time()
        for i in range(num_iterations):
            x_batch, y_true_batch = random_batch(x_train, y_train)
            session.run(optimizer, feed_dict={x: x_batch, y: y_true_batch, keep_prob: 0.5})
            if i % 100 == 0:
                acc = session.run(accuracy, feed_dict={x: x_batch, y: y_true_batch, keep_prob: 0.5})
                msg = "Optimization Iteration: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
                print(msg.format(i, acc))
        end_time = time.time()
        time_dif = end_time - start_time
        print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

    # if True, train 5 Neural network ##################################################################################
    if True:
        for i in range(num_networks):
            print("Neural network: {0}".format(i))
            x_train, y_train, _, _ = random_training_set()
            session.run(tf.global_variables_initializer())
            optimize(num_iterations=num_iterations, x_train=x_train, y_train=y_train)
            saver.save(sess=session, save_path=get_save_path(i))
            print()
    # if True, train 5 Neural network ##################################################################################

    def predict_cls(images, batch_size=256):
        num_images = len(images)
        pred_classes = np.zeros(shape=(num_images, ), dtype=np.float)
        # Now calculate the predicted classes for the batches.
        # We will just iterate through all the batches.
        i = 0 # The starting index for the next batch is denoted i.
        while i < num_images:
            j = min(i + batch_size, num_images) # The ending index for the next batch is denoted j.
            feed_dict = {x: images[i:j, :], keep_prob: 1.0}
            pred_classes[i:j] = session.run(y_pred, feed_dict=feed_dict)
            # Set the start-index for the next batch to the
            # end-index of the current batch.
            i = j
        return pred_classes

    # Calculate a boolean array whether the predicted cls for the images are correct and accuracy.
    def prediction_classes_boolean_accuracy(images, cls_true):
        pred_classes = predict_cls(images=images)
        prediction_boolean = (cls_true == pred_classes)
        prediction_accuracy = prediction_boolean.mean()
        return pred_classes, prediction_boolean, prediction_accuracy

    # Calculate a boolean array whether the images in the test-set are classified correctly.
    def test_prediction_classes_boolean_accuracy():
        return prediction_classes_boolean_accuracy(images=data.test.images,
                                           cls_true=data.test.cls)

    # Calculate a boolean array whether the images in the validation-set are classified correctly.
    def validation_prediction_classes_boolean_accuracy():
        return prediction_classes_boolean_accuracy(images=data.validation.images,
                                           cls_true=data.validation.cls)

    # ensemble_predictions #############################################################################################
    def ensemble_predictions():
        test_pred_cls = [] # predicted classes for images=data.test.images
        test_accuracies = []
        val_accuracies = []

        # For each neural network in the ensemble.
        for i in range(num_networks):
            # Reload the variables into the TensorFlow graph.
            saver.restore(sess=session, save_path=get_save_path(i))

            # Calculate the classification accuracy on the test-set.
            test_pred, _, test_acc = test_prediction_classes_boolean_accuracy()
            test_accuracies.append(test_acc)
            test_pred_cls.append(test_pred)

            # Calculate the classification accuracy on the validation-set.
            _, _, val_acc = validation_prediction_classes_boolean_accuracy()
            val_accuracies.append(val_acc)

            # Print status message.
            msg = "Network: {0}, Accuracy on Validation-Set: {1:.4f}, Test-Set: {2:.4f}"
            print(msg.format(i, val_acc, test_acc))

        return np.array(test_pred_cls), \
               np.array(test_accuracies), \
               np.array(val_accuracies)
    # ensemble_predictions #############################################################################################


    test_pred_cls, test_accuracies, val_accuracies = ensemble_predictions()
    print("Mean test-set accuracy: {0:.4f}".format(np.mean(test_accuracies)))
    print("Min test-set accuracy:  {0:.4f}".format(np.min(test_accuracies)))
    print("Max test-set accuracy:  {0:.4f}".format(np.max(test_accuracies)))
    # The predicted labels of the ensemble is a 3-dim array,
    # the first dim is the network-number,
    # the second dim is the image-number,
    # the third dim is the classification vector.
    print(test_pred_cls.shape)