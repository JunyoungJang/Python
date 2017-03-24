import tensorflow as tf

a = tf.to_int64(3.14)
b = tf.to_int64([0.7, 0.3, -1.5, -10.5, 3.14])

with tf.Session() as sess:
    print sess.run(a)
    print sess.run(b)
