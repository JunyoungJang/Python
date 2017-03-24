import tensorflow as tf

a = tf.constant([2.], dtype=tf.float32, name='Constant_a')
b = tf.constant([3.], dtype=tf.float32, name='Constant_b')
c = tf.constant([4.], dtype=tf.float32, name='Constant_c')

result = a + b * c

with tf.Session() as Simple_Operation:
    output = Simple_Operation.run(result)
    print output
