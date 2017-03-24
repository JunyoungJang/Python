import tensorflow as tf
import numpy as np

# tf.linspace(start_point [float], end_point_included [float], number_points_generated [int])
a = tf.linspace(1.0, 10.0, 4)

# np.arange or range (start_point, end_point_excluded, jump_size)
b1 = np.arange(1, 10, 4)    # both float and int are allowed
b2 = np.arange(1., 10., 4.) # both float and int are allowed
c = range(5)                # int only
d = range(0, 5)             # int only
e = range(0, 5, 1)          # int only

with tf.Session() as sess:
    print sess.run(a)
    print b1, b2
    print c, d, e



