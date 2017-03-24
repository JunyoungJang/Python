import tensorflow as tf

boolean_1 = tf.constant([[True , True], [False, False]])
boolean_2 = tf.constant([[[True, False], [True, False]], [[False, True], [False, True]], [[False, False], [False, True]]])

# tf.where return the locations of True values, so that I could use the result's shape[0] to get the number of Trues
a = tf.where(boolean_1)
b = tf.where(boolean_2)

with tf.Session() as sess:
    print sess.run(a)
    print sess.run( tf.shape(a)[0] )
    print sess.run(b)
    print sess.run(tf.shape(b)[0])
