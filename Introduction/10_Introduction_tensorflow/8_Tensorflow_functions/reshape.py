import tensorflow as tf

A = [1, 2, 3, 4, 5, 6, 7, 8, 9]

a = tf.reshape(A, shape=(3, 3))

with tf.Session() as sess:
    print sess.run(a)