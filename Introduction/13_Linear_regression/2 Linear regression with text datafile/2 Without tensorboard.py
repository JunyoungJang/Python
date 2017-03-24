import tensorflow as tf
import numpy as np
import pandas as pd

LEARNING_RATE = 0.1
EPOCH = 100
DISPLAY_STEP = 2

INPUT_SIZE = 3
OUTPUT_SIZE = 3

df = pd.read_csv('train_data_linear_regression.txt', sep='\s+')
df = np.array(df, np.float32)
x_data = df[:, 0:-1]
y_data = np.expand_dims( df[:, -1], axis=1)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

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