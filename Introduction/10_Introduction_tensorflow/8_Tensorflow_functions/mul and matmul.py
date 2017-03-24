import tensorflow as tf

A = [[2., 1.]]
B = [[2.],[2.]]

a = tf.mul(A, B)
b = tf.matmul(A, B)

with tf.Session() as sess:
    print sess.run(a)
    print sess.run(b)