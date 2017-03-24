import tensorflow as tf
import matplotlib.pyplot as plt

a = tf.random_normal([2, 100], mean=-10, stddev=4, seed=1)

with tf.Session() as sess:
    a_eval = sess.run(a)
    plt.plot(a_eval[0,:], a_eval[1,:], 'o')
    plt.show()
