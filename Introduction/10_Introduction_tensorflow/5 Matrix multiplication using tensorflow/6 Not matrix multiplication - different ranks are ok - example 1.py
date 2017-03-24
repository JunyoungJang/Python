import tensorflow as tf

a = tf.constant([[2., 2., 2.]], dtype=tf.float32)                             # rank 2
b = tf.constant([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]], dtype=tf.float32) # rank 2
c = tf.constant([1.], dtype=tf.float32)                                       # rank 1
d = tf.matmul(a, b) + c                                                       # rank 2


a_rank = tf.rank(a)
b_rank = tf.rank(b)
c_rank = tf.rank(c)
d_rank = tf.rank(d)

with tf.Session() as sess:
    a_run, b_run, c_run, d_run, a_rank_run, b_rank_run, c_rank_run, d_rank_run =sess.run([a, b, c, d, a_rank, b_rank, c_rank, d_rank])
    print a_run
    print b_run
    print c_run
    print d_run
    print a_rank_run, b_rank_run, c_rank_run, d_rank_run
