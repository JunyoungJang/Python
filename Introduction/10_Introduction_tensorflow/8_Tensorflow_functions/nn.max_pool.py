import tensorflow as tf

x = tf.constant([[[[1., 2., 3., 4.],
                 [2., 3., 4., 5.]]]])

a = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.Session() as sess:
    print sess.run(a)
