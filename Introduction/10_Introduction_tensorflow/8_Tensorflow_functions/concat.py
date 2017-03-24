import tensorflow as tf

c1 = tf.constant([1, 3, 5, 7, 9, 0, 2, 4, 6, 8])
c2 = tf.constant([1, 3, 5])

v1 = tf.constant([[1, 2, 3, 4, 5, 6], [7, 8, 9, 0, 1, 2]])
v2 = tf.constant([[1, 2, 3], [7, 8, 9]])

a = tf.concat(0, [c1, c2])
b = tf.concat(1, [v1, v2])

with tf.Session() as sess:
    print sess.run(a)
    print sess.run(b)
