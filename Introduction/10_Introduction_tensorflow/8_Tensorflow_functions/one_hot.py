import tensorflow as tf

# tf.one_hot(indices_where_we_put_one_hot, depth)
a = tf.one_hot([0, 1, 2, 1], 5)
b = tf.one_hot([3, 2, 1, 0, 1], 10)

with tf.Session() as sess:
    print sess.run(a)
    print sess.run(b)