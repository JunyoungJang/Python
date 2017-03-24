import tensorflow as tf
import matplotlib.pyplot as plt

# tf.truncated_normal
# These values are similar to values from a tf.random_normal
# except that values more than two standard deviations from the mean are discarded and re-drawn.
# This is the recommended initializer for neural network weights and filters.
a = tf.truncated_normal(shape=(2, 1000), mean=0, stddev=1, seed=1)

with tf.Session() as sess:
    a_eval = sess.run(a)
    plt.plot(a_eval[0,:], a_eval[1,:], 'o')
    plt.axis([-3,3,-3,3])
    plt.show()

