#-*- coding: utf-8 -*-
import tensorflow as tf

v = tf.constant([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])

# tf.pad(input, [[top_pad_size, down_pad_size], [left_pad_size, right_pad_size]], pad_mathod)
a = tf.pad(v, [[1, 2], [3, 4]], 'CONSTANT')
b = tf.pad(v, [[1, 2], [3, 4]], 'REFLECT')
c = tf.pad(v, [[1, 2], [3, 4]], 'SYMMETRIC')

with tf.Session() as sess:
    print sess.run(a)
    print sess.run(b)
    print sess.run(c)

