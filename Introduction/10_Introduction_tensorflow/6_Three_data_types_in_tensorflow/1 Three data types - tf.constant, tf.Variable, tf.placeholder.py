import tensorflow as tf

a = tf.constant(    [1, 2, 3], dtype=tf.float32)
b = tf.Variable(    [1, 2, 3], dtype=tf.float32, name='variable_b')
c = tf.placeholder(            dtype=tf.float32)

print a
print b
print c
