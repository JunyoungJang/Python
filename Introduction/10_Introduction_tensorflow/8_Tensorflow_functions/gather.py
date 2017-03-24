import tensorflow as tf

u = tf.constant([1, 3, 5, 7, 9, 0, 2, 4, 6, 8])
v = tf.constant([[1, 2, 3, 4, 5, 6], [7, 8, 9, 0, 1, 2]])

a = tf.gather(u, [2, 5, 2, 5]) # [2nd item, 5th item, 2nd item, 5th item]
b = tf.gather(v, [1, 0]) # [1st item, 0th item]
c = tf.gather(v, [[0, 0], [1, 1]]) # [[0th item, 0th item], [1st item, 1st item]]

with tf.Session() as sess:
    print sess.run(a)
    print sess.run(b)
    print sess.run(c)
