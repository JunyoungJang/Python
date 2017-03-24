import tensorflow as tf

X = [[2., 2.], [-3., 3.]]

a = tf.reduce_sum(X)
b = tf.reduce_sum(X, 0)
c = tf.reduce_sum(X, 1)

d = tf.reduce_mean(X)
e = tf.reduce_mean(X, 0)
f = tf.reduce_mean(X, 1)

with tf.Session() as sess:
    print sess.run(a)
    print sess.run(b)
    print sess.run(c)
    print sess.run(d)
    print sess.run(e)
    print sess.run(f)
