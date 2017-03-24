import tensorflow as tf

u = tf.constant([1., 2.])
v = tf.constant([1., 3.])

a = tf.less(u, v)
b = tf.less_equal(u, v)

with tf.Session() as sess:
    print sess.run(a)   # [False  True]
    print sess.run(b)   # [ True  True]
