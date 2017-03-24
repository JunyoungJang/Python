import tensorflow as tf

a = tf.constant(0.1)
b = tf.constant(0.1, shape=(2, ))
c = tf.constant(0.1, shape=(2,2))

with tf.Session() as sess:
    print sess.run(a)
    print sess.run(b)
    print sess.run(c)
