import tensorflow as tf

u = tf.constant([1, 3, 5])
v = tf.constant([[1, 2, 3], [7, 8, 9]])

a = tf.tile(u, [3])
b = tf.tile(v, [2, 2])

with tf.Session() as sess:
    print sess.run( tf.shape(a) )
    print sess.run( tf.shape(b) )


