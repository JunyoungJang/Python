import tensorflow as tf

LEARNING_RATE = 0.1
LEARNING_RATE_2 = 0.1 # This line is changed -------------------------------------------------------------------------------------
EPOCH = 100
DISPLAY_STEP = 10
LOG_DIR = "./temp/logfile"
SAVE_PATH = LOG_DIR + "/savefile.ckpt"


# data
x_data = [[1.], [2.], [3.]]
y_data = [[1.], [2.], [3.]]

# placeholder
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# parameter
w = tf.Variable(tf.random_uniform((1,1), -1.0, 1.0), tf.float32, name='weight')
b = tf.Variable(tf.zeros((1,1)), tf.float32, name='bias')
parameter_list = [w, b]
saver = tf.train.Saver(parameter_list)

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
    x_data_new = [[5.]]
    y_data_predicted = Linear_regression.run(hypothesis, feed_dict={x: x_data_new})
    print 'New input : ', x_data_new
    print 'Predicted output : ', y_data_predicted

    x_data_new = [[5.], [-1.], [4.]]
    y_data_predicted = Linear_regression.run(hypothesis, feed_dict={x: x_data_new})
    print 'New input : ', x_data_new
    print 'Predicted output : ', y_data_predicted

    saver.save(Linear_regression, SAVE_PATH)

print "======================================================== Restoration ==========================================="

# data
x_2_data = [[1.], [2.], [3.]] # This line is changed -----------------------------------------------------------------------------
y_2_data = [[1.], [2.], [3.]] # This line is changed -----------------------------------------------------------------------------

# placeholder
x_2 = tf.placeholder(dtype=tf.float32) # This line is changed --------------------------------------------------------------------
y_2 = tf.placeholder(dtype=tf.float32) # This line is changed --------------------------------------------------------------------
feed_info_2 = {x_2:x_2_data, y_2:y_2_data} # This line is changed ----------------------------------------------------------------

# parameter
w_2 = tf.Variable(tf.random_uniform((1,1), -1.0, 1.0), dtype=tf.float32, name='weight_2') # This line is changed -----------------
b_2 = tf.Variable(tf.zeros((1,1)), dtype=tf.float32, name='bias_2') # This line is changed ---------------------------------------

# model
hypothesis_2 = x_2 * w_2 + b_2 # This line is changed ----------------------------------------------------------------------------

# find optimal parameter
with tf.name_scope('Train_2') as Unit_2: # This line is changed ------------------------------------------------------------------
    cost_2 = tf.reduce_mean( tf.square( hypothesis_2 - y_2 ) ) # This line is changed --------------------------------------------
    train_2 = tf.train.GradientDescentOptimizer(LEARNING_RATE_2).minimize(cost_2) # This line is changed -------------------------

# cd Dropbox/Tensorflow/13\ Linear regression/2\ Linear regression/temp/Linear regression logs/
# tensorboard --logdir=./
# http://localhost:6006/
