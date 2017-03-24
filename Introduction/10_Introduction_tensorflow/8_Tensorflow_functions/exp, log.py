import tensorflow as tf

X = tf.constant([[2., 2.], [-3., 3.]])

a = tf.exp(X)
b = tf.log(tf.abs(X))

with tf.Session() as sess:
    print sess.run(a)
    print sess.run(b)
