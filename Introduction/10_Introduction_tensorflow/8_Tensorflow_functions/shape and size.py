import tensorflow as tf
import numpy as np

E = np.array([[31, 23,  4, 24, 27, 34],
              [18,  3, 25,  0,  6, 35],
              [28, 14, 33, 22, 20,  8],
              [13, 30, 21, 19,  7,  9],
              [16,  1, 26, 32,  2, 29],
              [17, 12,  5, 11, 10, 15]])

a = tf.shape(E)
b = tf.size(E)

with tf.Session() as sess:
    print sess.run(a)
    print sess.run(b)
