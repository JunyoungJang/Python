import tensorflow as tf
import numpy as np

LEARNING_RATE = 0.1
EPOCH = 100
DISPLAY_STEP = 2

INPUT_SIZE = 3
OUTPUT_SIZE = 3

xy = np.loadtxt('train_data_linear_regression.txt', unpack=True, dtype='float32')
x_data = xy[0:-1] # "unpack=True" : read 0:-1 columns excluding the last column, take transpose, and report in row form
print x_data
x_data = np.transpose(x_data)
print x_data
y_data = np.expand_dims(xy[-1], axis=0) # "unpack=True" : read the last column, take transpose, and report in row form
print y_data
y_data = np.transpose(y_data)
print y_data

X = tf.placeholder(dtype=tf.float32)
Y = tf.placeholder(dtype=tf.float32)

W = tf.Variable(tf.random_uniform((INPUT_SIZE, OUTPUT_SIZE), -1., 1.), dtype=tf.float32, name='weight')

hypothesis = tf.matmul(X, W)

cost = tf.reduce_mean(tf.square(hypothesis - Y))
train = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for step in range(EPOCH):
        if step % DISPLAY_STEP != 0:
            sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % DISPLAY_STEP == 0:
            _, cost_now, W_now = sess.run([train, cost, W], feed_dict={X: x_data, Y: y_data})
            print step, cost_now, W_now
