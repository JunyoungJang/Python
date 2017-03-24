import tensorflow as tf
import numpy as np

a = tf.constant(    [1, 2, 3], dtype=tf.float32)
b = tf.Variable(    [1, 2, 3], dtype=tf.float32, name='variable_b')
c = tf.placeholder(            dtype=tf.float32)
d = c + 1

print a
print b
print c
print d

with tf.Session() as Simple_Operation:

    result_1 = Simple_Operation.run(a)
    print result_1

    # to run an expression containing tf.Variable, we need to either initialize or restore
    tf.global_variables_initializer().run()
    result_2 = Simple_Operation.run(b)
    print result_2

    # to run an expression containing tf.placeholder, we need to feed a specific value
    result_3 = Simple_Operation.run(d, feed_dict={c: [1, 2, 3]})
    print result_3
