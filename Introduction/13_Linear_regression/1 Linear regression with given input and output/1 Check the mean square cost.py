import tensorflow as tf

x_data = [[1.], [2.], [3.]]
y_data = [[1.], [2.], [3.]]

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

w = tf.Variable([[2.]], tf.float32, name='weight')
b = tf.Variable([[2.]], tf.float32, name='bias')

hypothesis = tf.matmul(x, w) + b
cost = tf.reduce_mean(tf.square(hypothesis - y))

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    input, output, loss = sess.run([x, hypothesis, cost], feed_dict={x:x_data, y:y_data})
    print 'Input x\n', input
    print 'Output y\n', output
    print 'Loss\n', loss
