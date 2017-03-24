import tensorflow as tf

a = tf.fill([2,3], 2)
b = tf.fill([2,3], 2.0)

with tf.Session() as sess:
    print sess.run(a)
    print sess.run(b)
