import tensorflow as tf

A = tf.random_normal([5, 30])

# tf.slice(axis_to_split, numbers_of_pieces_after_splitting, input)
split_0, split_1, split_2, split_3, split_4, split_5 = tf.split(1, 6, A)
SPLIT_0, SPLIT_1, SPLIT_2 = tf.split(1, 3, A)

with tf.Session() as sess:
    print sess.run( tf.shape(split_0) )
    print sess.run( tf.shape(split_1) )
    print sess.run( tf.shape(split_2) )
    print sess.run( tf.shape(SPLIT_0) )
    print sess.run( tf.shape(SPLIT_1) )
    print sess.run( tf.shape(SPLIT_2) )
