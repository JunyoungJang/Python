import tensorflow as tf

u = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
v = tf.constant([[1, 2, 3, 4, 5, 6], [7, 8, 9, 0, 1, 2]])

# tf.slice(input, begin_spot, size_to_slice)
a = tf.slice(u, [2], [3])
b = tf.slice(v, [0, 2], [1, 2])
c = tf.slice(v, [0, 2], [2, 2])
d = tf.slice(v, [0, 2], [2,-1])

with tf.Session() as sess:
    print sess.run(a)
    print sess.run(b)
    print sess.run(c)
    print sess.run(d)
