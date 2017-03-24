import tensorflow as tf

X = tf.constant([[2., 2.], [-3., 3.]])
Y = tf.constant([[2., -4.], [2., 3.]])

# tf.sub(X, Y) is equivalent to X - Y
a = tf.sub(X, Y)
b = X - Y

with tf.Session() as sess:
    print sess.run(a)
    print sess.run(b)
