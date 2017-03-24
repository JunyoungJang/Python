import tensorflow as tf

logit = tf.constant([[2., 1.]])
label = tf.constant([[1., 0.]])
label_argmax = tf.argmax(label, 1)

a = tf.nn.softmax_cross_entropy_with_logits(logit, label)
b = tf.nn.sparse_softmax_cross_entropy_with_logits(logit, label_argmax)

prob = tf.nn.softmax(logit)
log_prob = tf.log(prob)
c = - tf.reduce_sum( tf.mul(label, log_prob) )

with tf.Session() as sess:
    print sess.run(a)
    print sess.run(b)
    print sess.run(c)

