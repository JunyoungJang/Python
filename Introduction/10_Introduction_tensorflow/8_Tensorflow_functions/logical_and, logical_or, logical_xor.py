import tensorflow as tf
import numpy as np

u = tf.constant([[True , True], [False, False]])
v = tf.constant([[True, False], [True, False]])

a = tf.logical_and(u, v)
b = tf.logical_or(u, v)
c = tf.logical_xor(u, v)

with tf.Session() as sess:
    print sess.run(a)
    print sess.run(b)
    print sess.run(c)

