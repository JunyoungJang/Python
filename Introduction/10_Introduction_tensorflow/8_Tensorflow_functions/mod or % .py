import tensorflow as tf
import numpy as np

X = tf.constant([[1., 2.], [3., 5.]])

# tf.mod(X, 2) is equivalent to X % 2
a = tf.mod(X, 2)
b = X % 2

with tf.Session() as sess:
    print sess.run(a)
    print sess.run(b)
