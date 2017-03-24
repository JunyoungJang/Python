import tensorflow as tf

LEARNING_RATE = 0.1
EPOCH = 100
DISPLAY_STEP = 10
LOG_DIR = "./temp/logfile"

# data
x_data = [[1.], [2.], [3.]]
y_data = [[1.], [2.], [3.]]

# placeholder
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# parameter
w = tf.Variable(tf.random_uniform((1,1), -1.0, 1.0), tf.float32, name='weight')
b = tf.Variable(tf.zeros((1,1)), tf.float32, name='bias')

# model
with tf.name_scope('Hypothesis') as Unit_1:
    hypothesis = x * w + b

# find optimal parameter
with tf.name_scope('Train') as Unit_2:
    cost = tf.reduce_mean(tf.square(hypothesis - y))
    train = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

w_summ = tf.summary.histogram('weight_Record', w)
b_summ = tf.summary.histogram('bias_Record', b)
cost_summ = tf.summary.scalar('cost_Record', cost)
merged = tf.summary.merge_all()

with tf.Session() as Linear_regression:
    writer = tf.summary.FileWriter(LOG_DIR, Linear_regression.graph)
    tf.global_variables_initializer().run()

    for step_index in range(EPOCH):
        if step_index % DISPLAY_STEP != 0:
            Linear_regression.run(train, feed_dict={x:x_data, y:y_data})
        if step_index % DISPLAY_STEP == 0:
            _, cost_now, w_now, b_now, merged_now = Linear_regression.run([train, cost, w, b, merged], feed_dict={x:x_data, y:y_data})
            writer.add_summary(merged_now, step_index)
            print '========================================='
            print 'Step : ', step_index
            print 'Cost : ', cost_now
            print 'Weight : ', w_now
            print 'Bias : ', b_now

    # predict output for a new input data
    x_data_new = [[5.]] # This line is changed -----------------------------------------------------------------------------------
    y_data_predicted = Linear_regression.run(hypothesis, feed_dict={x: x_data_new}) # This line is changed -----------------------
    print 'New input : ', x_data_new # This line is changed ----------------------------------------------------------------------
    print 'Predicted output : ', y_data_predicted # This line is changed ---------------------------------------------------------

    x_data_new = [[5.], [-1.], [4.]] # This line is changed ----------------------------------------------------------------------
    y_data_predicted = Linear_regression.run(hypothesis, feed_dict={x: x_data_new}) # This line is changed -----------------------
    print 'New input : ', x_data_new # This line is changed ----------------------------------------------------------------------
    print 'Predicted output : ', y_data_predicted # This line is changed ---------------------------------------------------------

# cd Dropbox/Tensorflow/13\ Linear regression/2\ Linear regression/temp/Linear regression logs/
# tensorboard --logdir=./
# http://localhost:6006/
