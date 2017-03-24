import tensorflow as tf

A = tf.constant([[2., 1.]])

a = tf.nn.softmax(A)
b = tf.exp(A) / tf.reduce_sum(tf.exp(A))

with tf.Session() as sess:
    print sess.run(a)
    print sess.run(b)
