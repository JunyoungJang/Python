import tensorflow as tf

X = tf.constant([[2., 2.], [-3., 3.]])
Y = tf.constant([[2., -4.], [2., 3.]])

a = tf.maximum(X, Y)
b = tf.minimum(X, Y)

with tf.Session() as sess:
    print sess.run(a)
    print sess.run(b)
