import tensorflow as tf

X = tf.constant([[2., 2.], [-3., 3.]])

a = tf.round(X)
b = tf.ceil(X)
c = tf.floor(X)

with tf.Session() as sess:
    print sess.run(a)
    print sess.run(b)
    print sess.run(c)
