import tensorflow as tf

X = tf.constant([[2, 2.], [-3, 0]])

a = tf.sign(X)

with tf.Session() as sess:
    print sess.run(a)