import tensorflow as tf
import matplotlib.pyplot as plt

a = tf.random_uniform((2, 100), 0, 1, seed=0)

with tf.Session() as sess:
    a_eval = sess.run(a)
    plt.plot(a_eval[0,:], a_eval[1,:], 'o')
    plt.axis([0,1,0,1])
    plt.show()
