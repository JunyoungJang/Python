import tensorflow as tf

LEARNING_RATE = 0.1

# data
x_data = [[1.], [2.], [3.]]
y_data = [[1.], [2.], [3.]]

# parameter
w = tf.Variable(tf.random_uniform((1,1), -1.0, 1.0), tf.float32, name='weight')
b = tf.Variable(tf.zeros((1,1)), tf.float32, name='bias')

# model
hypothesis = x_data * w + b

# find optimal parameter
cost = tf.reduce_mean(tf.square(hypothesis - y_data))
train = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

with tf.Session() as Linear_regression:
    tf.global_variables_initializer().run()
    
    Linear_regression.run(train)
