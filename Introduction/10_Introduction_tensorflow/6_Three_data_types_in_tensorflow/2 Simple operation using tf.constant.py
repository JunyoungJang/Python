import tensorflow as tf

a = tf.constant([2.], dtype=tf.float32)
b = tf.constant([3.], dtype=tf.float32)
c = tf.constant([4.], dtype=tf.float32)

result = a + b * c

with tf.Session() as Simple_Operation:
    output = Simple_Operation.run(result)
    print output