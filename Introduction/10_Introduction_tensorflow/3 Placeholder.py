import tensorflow as tf

a = tf.placeholder(tf.double) # tf.placeholder works as a place reserved for a function argument
b = tf.placeholder(tf.double) # tf.placeholder works as a place reserved for a function argument
c = tf.placeholder(tf.double)
c = a + b

with tf.Session() as sess:
    print sess.run(c, feed_dict={a: 2, b: 3}) # You have to feed the function arguments as a dictionary type value
    print sess.run(c, {a: 2, b: 3})  # You have to feed the function arguments as a dictionary type value
