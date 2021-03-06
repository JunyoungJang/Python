import tensorflow as tf
import numpy as np

x1 = np.array([2, 2, 1, 0, 7])
x2 = np.array([2, 0, 1, 0, 7])

a = tf.equal(x1, x2)
b = tf.not_equal(x1, x2)

with tf.Session() as sess:
    print sess.run(a) # [ True False  True  True  True]
    print sess.run(b) # [False  True False False False]


