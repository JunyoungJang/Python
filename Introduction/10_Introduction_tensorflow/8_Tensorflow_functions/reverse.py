import tensorflow as tf

u = tf.constant([1, 2, 3, 4, 5, 6])
v = tf.constant([[1, 2, 3, 4, 5, 6], [7, 8, 9, 0, 1, 2]])

a = tf.reverse(u, [True])
b = tf.reverse(v, [True, False])
c = tf.reverse(v, [True, True])

with tf.Session() as sess:
    print sess.run(a)
    print sess.run(b)
    print sess.run(c)
