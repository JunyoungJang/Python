import tensorflow as tf

X = tf.constant([[2., 2.], [-3., 3.]])
Y = tf.constant([[2., -4.], [2., 3.]])
A = tf.constant([[2., 1.]])

# tf.add(X, Y) is equivalent to X + Y
a = tf.add(X, Y)
b = X + Y
c = tf.add(X, A)
d = X + A

with tf.Session() as sess:
    print sess.run(a)
    print sess.run(b)
    print sess.run(c)
    print sess.run(d)

