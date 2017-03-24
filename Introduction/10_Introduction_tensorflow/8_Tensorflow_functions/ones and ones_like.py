import tensorflow as tf

A = tf.constant([[1, 2, 3], [4, 5, 6]])

a = tf.ones([2, 3])
b = tf.ones_like(A)

with tf.Session() as sess:
    print sess.run(a)
    print sess.run(b)

