import tensorflow as tf

a = tf.constant([[2., 2., 2.]], dtype=tf.float32)                             # rank 2
b = tf.constant([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]], dtype=tf.float32) # rank 2
c = tf.matmul(a, b)                                                           # rank 2

a_rank = tf.rank(a)
b_rank = tf.rank(b)
c_rank = tf.rank(c)

with tf.Session() as sess:
    a_run, b_run, c_run, a_rank_run, b_rank_run, c_rank_run =sess.run([a, b, c, a_rank, b_rank, c_rank])
    print a_run
    print b_run
    print c_run
    print a_rank_run, b_rank_run, c_rank_run
