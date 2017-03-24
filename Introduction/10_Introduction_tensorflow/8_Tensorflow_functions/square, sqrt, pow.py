import tensorflow as tf

X = [[2., 2.], [-3., 3.]]
Y = [[2., -4.], [2., 3.]]

a = tf.square(X)
b = tf.sqrt( tf.abs(X) )
c = tf.pow(X, 2)
d = tf.pow(X, 3)
e = tf.pow(X, Y)

with tf.Session() as sess:
    print sess.run(a)
    print sess.run(b)
    print sess.run(c)
    print sess.run(d)
    print sess.run(e)
