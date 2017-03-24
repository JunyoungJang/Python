import tensorflow as tf
import numpy as np

X = tf.constant([[2., 2.], [-3., 3.]])
Y = tf.constant([[2., -4.], [2., 3.]])

# tf.div(x, y) is equivalent to x / y
a = tf.div(X, Y)
b = X / Y

with tf.Session() as sess:
    print sess.run(a)
    print sess.run(b)
