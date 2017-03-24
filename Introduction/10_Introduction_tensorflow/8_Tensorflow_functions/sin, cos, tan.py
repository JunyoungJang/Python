import tensorflow as tf

X = tf.constant([[2., 2.], [-3., 3.]])

a = tf.sin(X)
b = tf.cos(X)
c = tf.tan(X)

with tf.Session() as sess:
    print sess.run(a)
    print sess.run(b)
    print sess.run(c)