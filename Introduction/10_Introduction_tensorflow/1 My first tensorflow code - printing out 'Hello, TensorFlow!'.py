import tensorflow as tf

a = tf.constant('Hello, TensorFlow!')

with tf.Session() as sess:
    print sess.run(a)
