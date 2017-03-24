import numpy as np

X = np.array([[2., 2.], [-3., 3.]])
print np.abs(X)

import tensorflow as tf

X = tf.constant([[2., 2.], [-3., 3.]])
a = tf.abs(X)
with tf.Session() as sess:
    print sess.run(a)
