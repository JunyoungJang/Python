import tensorflow as tf

a = tf.random_shuffle([1, 2, 3, 4, 5, 6])

with tf.Session() as sess:
    print sess.run(a)

