import tensorflow as tf

A = tf.constant([[2., 1.]])
X = tf.constant([[2., 2.], [-3., 3.]])

# Contrast to python, taking the transpose of a rank 1 array using tf.transpose works properly in mathmatical sense.
a = tf.transpose(A)
x = tf.transpose(X)

with tf.Session() as sess:
    print sess.run(a)
    print sess.run(x)
