import tensorflow as tf

u1, u2 = tf.constant([1, 3, 5]), tf.constant([[1, 3, 5], [5, 7, 9]])
v1, v2 = tf.constant([2, 4, 6]), tf.constant([[2, 4, 6], [6, 8, 0]])

a = tf.pack([u1, v1])
b = tf.pack([u2, v2])
c = tf.unpack(a)
d = tf.unpack(b)

with tf.Session() as sess:
    print sess.run(a)
    print sess.run(b)
    print sess.run(c)
    print sess.run(d)

