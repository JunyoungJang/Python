import tensorflow as tf

X = tf.constant([[2., 2.], [-3., 3.]])

# tf.neg(X) is equivalent to - X
a = tf.neg(X)
b = - X

with tf.Session() as sess:
    print sess.run(a)
    print sess.run(b)
