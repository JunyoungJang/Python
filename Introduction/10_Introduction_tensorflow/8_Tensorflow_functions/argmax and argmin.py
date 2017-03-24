import tensorflow as tf

A = [[31, 23],
     [18, 19],
     [17, 22]]

a = tf.argmax(A, 0)
b = tf.argmax(A, 1)


B = [1, 0, 1, 2, 3]

c = tf.argmin(B, 0)
# d = tf.argmin(B, 1) # InvalidArgumentError (see above for traceback): Minimum tensor rank: 2 but got: 1

with tf.Session() as sess:
    print sess.run(a)
    print sess.run(b)
    print sess.run(c)
    # print sess.run(d) # InvalidArgumentError (see above for traceback): Minimum tensor rank: 2 but got: 1




