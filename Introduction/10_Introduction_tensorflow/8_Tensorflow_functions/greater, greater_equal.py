import tensorflow as tf

u = tf.constant([1., 2.])
v = tf.constant([1., 3.])

a = tf.greater(u, v)
b = tf.greater_equal(u, v)

with tf.Session() as sess:
    print sess.run(a)   # [False False]
    print sess.run(b)   # [ True False]

