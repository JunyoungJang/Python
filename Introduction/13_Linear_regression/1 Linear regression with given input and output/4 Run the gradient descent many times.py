import tensorflow as tf

LEARNING_RATE = 0.1
EPOCH = 100 # This line is changed -----------------------------------------------------------------------------------------------
DISPLAY_STEP = 10 # This line is changed -----------------------------------------------------------------------------------------

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

    for step_index in range(EPOCH):  # This line is changed ----------------------------------------------------------------------
        if step_index % DISPLAY_STEP != 0:  # This line is changed ---------------------------------------------------------------
            Linear_regression.run(train)  # This line is changed -----------------------------------------------------------------
        if step_index % DISPLAY_STEP == 0:  # This line is changed ---------------------------------------------------------------
            _, cost_now, w_now, b_now = Linear_regression.run([train, cost, w, b])  # This line is changed -----------------------
            print '========================================='  # This line is changed --------------------------------------------
            print 'Step : ', step_index  # This line is changed ------------------------------------------------------------------
            print 'Cost : ', cost_now  # This line is changed --------------------------------------------------------------------
            print 'Weight : ', w_now  # This line is changed ---------------------------------------------------------------------
            print 'Bias : ', b_now  # This line is changed -----------------------------------------------------------------------
