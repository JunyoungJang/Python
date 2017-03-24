import tensorflow as tf

LEARNING_RATE = 0.1
EPOCH = 100
DISPLAY_STEP = 10
LOG_DIR = "./temp/logfile" # Where to save. This line is changed --------------------------------------------------------------------------------

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
    writer = tf.summary.FileWriter(LOG_DIR, Linear_regression.graph) # Action. This line is changed ------------------------------------
    tf.global_variables_initializer().run()

    for step_index in range(EPOCH):
        if step_index % DISPLAY_STEP != 0:
            Linear_regression.run(train)
        if step_index % DISPLAY_STEP == 0:
            _, cost_now, w_now, b_now = Linear_regression.run([train, cost, w, b])
            print '========================================='
            print 'Step : ', step_index
            print 'Cost : ', cost_now
            print 'Weight : ', w_now
            print 'Bias : ', b_now

# cd Dropbox/Tensorflow/13\ Linear regression/2\ Linear regression/temp/Linear regression logs/
# tensorboard --logdir=./
# http://localhost:6006/
