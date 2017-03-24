import tensorflow as tf

logit = tf.constant([[1.0, 0.7, 2.0, 0.5, 2.7, -6.3]])
label = tf.constant([[  0,   0,   1,   0,   0,    0]])
label_argmax = tf.argmax(label, 1)

a = tf.nn.in_top_k(logit, label_argmax, 1)  # [False]
b = tf.nn.in_top_k(logit, label_argmax, 2)  # [ True]
c = tf.nn.in_top_k(logit, label_argmax, 3)  # [ True]

with tf.Session() as sess:
    print sess.run(a)
    print sess.run(b)
    print sess.run(c)


