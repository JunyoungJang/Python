import tensorflow as tf

A = tf.constant([[1, 2, 3], [4, 5, 6]], tf.float32)

a = tf.zeros((2, 3))
b = tf.zeros_like(A)

with tf.Session() as sess:
    print sess.run(a)
    print sess.run(b)