import tensorflow as tf

a = tf.placeholder(dtype=tf.float32)
b = tf.placeholder(dtype=tf.float32)
c = tf.placeholder(dtype=tf.float32)

result = a + b * c

with tf.Session() as Simple_Operation:
    
    # to run an expression containing tf.placeholder, we need to feed a specific value
    output = Simple_Operation.run(result, feed_dict={a: [2.], b: [3.], c: [4.]})
    print output
