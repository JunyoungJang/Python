import tensorflow as tf

A = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

# Removes dimensions of size 1 from the shape of a tensor.
# Given a tensor input, this operation returns a tensor of the same type with all dimensions of size 1 removed.
# If you don't want to remove all size 1 dimensions, you can remove specific size 1 dimensions by specifying squeeze_dims.
a = tf.reshape(A, [4, 1, 3])
b = tf.squeeze(a)

with tf.Session() as sess:
    print sess.run( tf.shape(a) )
    print sess.run( tf.shape(b) )
