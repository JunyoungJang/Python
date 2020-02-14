# https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/06_CIFAR-10.ipynb

import tensorflow as tf
# from cs231n_utilities import load_CIFAR10
import numpy as np
import os
import _pickle as pickle
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import prettytensor as pt

LEARNING_RATE = 1e-3
DECAY_RATE = 0.9
EPOCH_TRAIN = 300
BATCH_SIZE = 128
TEST_SIZE = 256
img_size_cropped = 24
num_channels = 3
num_classes = 10
num_iterations = 1
train_batch_size = 64
momentum = 0.9
epsilon = 1e-06
save_dir = "./temp/logfile"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'savefile.ckpt')


# load cifar10 data ####################################################################################################
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

images_train, cls_train, images_test, cls_test = load_CIFAR10('/Users/sungchul/Dropbox/Data/CIFAR-10/')  # a magic function we provide
print(type(images_train), images_train.shape)
print(type(cls_train), cls_train.shape)
print(type(images_test), images_test.shape)
print(type(cls_test), cls_test.shape)
print(cls_test[0:20])

def one_hot_encode(Y, list_of_class_numbers):
    Y_one_hot = np.zeros((len(Y), len(list_of_class_numbers)))
    for i, class_number in enumerate(list_of_class_numbers):
        Y_one_hot[Y==class_number, i] = 1
    return Y_one_hot

labels_train = one_hot_encode(cls_train, list_of_class_numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
labels_test = one_hot_encode(cls_test, list_of_class_numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

class_names = ['airplane',
               'automobile',
               'bird',
               'cat',
               'deer',
               'dog',
               'frog',
               'horse',
               'ship',
               'truck']
# load cifar10 data ####################################################################################################


# Plot a few images to see if data is correct ##########################################################################
def plot_images(images, cls_true, cls_pred=None, smooth=True):
    assert len(images) == len(cls_true) == 9
    fig, axes = plt.subplots(3, 3) # Create figure with sub-plots.

    if cls_pred is None: # Adjust vertical spacing if we need to print ensemble and best-net.
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        if smooth: # Interpolation type.
            interpolation = 'spline16'
        else:
            interpolation = 'nearest'

        cls_true_name = class_names[cls_true[i]]  # Name of the true class.
        if cls_pred is None: # Show true classes only.
            xlabel = "True: {0}".format(cls_true_name)
        else: # Show true and predicted classes.
            cls_pred_name = class_names[cls_pred[i]] # Name of the predicted class.
            xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)

        ax.imshow(images[i, :, :, :],
                  interpolation=interpolation) # Plot image.
        ax.set_xlabel(xlabel) # Show the classes as the label on the x-axis.
        ax.set_xticks([]) # Remove ticks from the plot.
        ax.set_yticks([]) # Remove ticks from the plot.
    plt.show() # Ensure the plot is shown correctly with multiple plots in a single Notebook cell.


images = images_test[0:9]
cls_true = cls_test[0:9]
#plot_images(images=images, cls_true=cls_true, smooth=False)
plot_images(images=images, cls_true=cls_true, smooth=True)
# Plot a few images to see if data is correct ##########################################################################


# Plot a few distorted images ##########################################################################################
def pre_process_image(image, training):
    # This function takes a single image as input,
    # and a boolean whether to build the training or testing graph.
    if training: # Randomly crop the input image.
        image = tf.random_crop(image, size=[img_size_cropped, img_size_cropped, num_channels])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
        image = tf.image.random_brightness(image, max_delta=0.02)
        image = tf.image.random_saturation(image, lower=0.0, upper=2.0)
        image = tf.minimum(image, 256.0) # Limit the image pixels between [0, 256] in case of overflow.
        image = tf.maximum(image, 0.0) # Limit the image pixels between [0, 256] in case of overflow.
    else: # Crop the input image around the centre so it is the same size as train images.
        image = tf.image.resize_image_with_crop_or_pad(image,
                                                       target_height=img_size_cropped,
                                                       target_width=img_size_cropped)
    return image


def pre_process_images(images, training):
    # This function takes many images as input,
    # and a boolean whether to build the training or testing graph.
    images = tf.map_fn(lambda image: pre_process_image(image, training), images)
    return images

with tf.Session() as session:
    tf.global_variables_initializer().run()


    def plot_distorted_image(image, cls_true):
        image_duplicates = np.repeat(image[np.newaxis, :, :, :], 9, axis=0)  # Repeat the input image 9 times.
        distorted_images = session.run(pre_process_images(image_duplicates, training=True))
        plot_images(images=distorted_images, cls_true=np.repeat(cls_true, 9))


    image = images_test[0]
    cls_true = cls_test[0]
    plot_test = plot_distorted_image(image, cls_true)
# Plot a few distorted images ##########################################################################################


# Create Neural Network for Training Phase #############################################################################
def weight_variable(shape):
    return tf.Variable(0.1*tf.truncated_normal(shape=shape, stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder(tf.float32)
x0 = pre_process_images(x, training=True)
y = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)

# 64 filters of shape (5,5,3)
# input shape (-1,24, 24, 3)
# output shape (-1,12,12,64)
W_conv1 = weight_variable(shape=(5, 5, 3, 64))
b_conv1 = bias_variable(shape=(1, 64))
h_conv1 = tf.nn.relu(conv2d(x0, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 64 filters of shape (5,5,64)
# input shape (-1,12,12,64)
# output shape (-1,6,6,64)
W_conv2 = weight_variable(shape=(5, 5, 64, 64))
b_conv2 = bias_variable(shape=(1, 64))
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# input shape (-1,6,6,64)
# output shape (-1,6*6*64)
h_pool2_flat = tf.reshape(h_pool2, shape=(-1, 6*6*64))

# input shape (-1,6*6*64)
# output shape (-1,256)
W_fc1 = weight_variable(shape=(6*6*64, 256))
b_fc1 = bias_variable(shape=(1, 256))
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# input shape (-1,256)
# output shape (-1,128)
W_fc2 = weight_variable(shape=(256, 128))
b_fc2 = bias_variable(shape=(1, 128))
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

# input shape (-1,128)
# output shape (-1,10)
W_fc3 = weight_variable(shape=(128, 10))
b_fc3 = bias_variable(shape=(1, 10))
y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
train = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost, global_step=global_step)
prediction = tf.argmax(y_conv,1)
correct_prediction = tf.equal(prediction, tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Create Neural Network for Training Phase #############################################################################


# Saver
saver = tf.train.Saver()


# Create TensorFlow session
with tf.Session() as session:

    try:
        print("Trying to restore last checkpoint ...")
        last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
        saver.restore(session, save_path=last_chk_path)
        print("Restored checkpoint from:", last_chk_path)
    except:
        print("Failed to restore checkpoint. Initializing variables instead.")
        session.run(tf.global_variables_initializer())


    def random_batch():
        num_images = len(images_train)
        idx = np.random.choice(num_images,
                               size=train_batch_size,
                               replace=False)
        x_batch = images_train[idx, :, :, :]
        y_batch = labels_train[idx, :]
        return x_batch, y_batch

    def optimize(num_iterations):
        start_time = time.time()

        for i in range(num_iterations):
            x_batch, y_true_batch = random_batch()
            feed_dict_train = {x: x_batch, y: y_true_batch, keep_prob: 0.5}
            i_global, _ = session.run([global_step, train], feed_dict=feed_dict_train)

            if (i_global % 100 == 0) or (i == num_iterations - 1):
                batch_acc = session.run(accuracy, feed_dict=feed_dict_train)
                print("Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%}".format(i_global, batch_acc))

            if (i_global % 1000 == 0) or (i == num_iterations - 1):
                saver.save(session, save_path=save_path, global_step=global_step)
                print("Saved checkpoint.")

        end_time = time.time()
        time_dif = end_time - start_time
        print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


    if True:
        optimize(num_iterations=5000)

    # test_image = images_test[0]
    # test_image_cropped = session.run(pre_process_image(image=test_image, training=False))
    # feed_dict_test = {x0: [test_image_cropped], y: labels_test[0], keep_prob: 1.0}
    # pred = session.run(prediction, feed_dict_test)
    # print(cls_test[0])
    # print(pred)

    test_images = images_test[0:10]
    test_images_cropped = session.run(pre_process_images(images=test_images, training=False))
    feed_dict_test = {x0: test_images_cropped, y: labels_test[0:10], keep_prob: 1.0}
    pred = session.run(prediction, feed_dict_test)
    print(cls_test[0:10])
    print(pred)


# Create Neural Network for Test Phase #################################################################################
# y_pred, _ = create_network(training=False)
# y_pred_cls = tf.argmax(y_pred, dimension=1)
# correct_prediction = tf.equal(y_pred_cls, y_true_cls)
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Create Neural Network for Test Phase #################################################################################