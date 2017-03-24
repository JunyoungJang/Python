import tensorflow as tf

LEARNING_RATE = 0.1
LEARNING_RATE_2 = 0.1
EPOCH = 100
EPOCH_2 = 100
DISPLAY_STEP = 10
DISPLAY_STEP_2 = 10
LOG_DIR = "./temp/logfile"
SAVE_PATH = LOG_DIR + "/savefile.ckpt"
SAVE_PATH_2 = LOG_DIR + "/savefile2.ckpt" # This line is changed -----------------------------------------------------------------

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
            _, cost_now, w_now, b_now, merged_now = Linear_regression.run([train, cost, w, b, merged],
                                                                          feed_dict={x:x_data, y:y_data})
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
x_2_data = [[1.], [2.], [3.]]
y_2_data = [[1.], [2.], [3.]]

# placeholder
x_2 = tf.placeholder(dtype=tf.float32)
y_2 = tf.placeholder(dtype=tf.float32)
feed_info_2 = {x_2:x_2_data, y_2:y_2_data}

# parameter
w_2 = tf.Variable(tf.random_uniform((1,1), -1.0, 1.0), dtype=tf.float32, name='weight_2')
b_2 = tf.Variable(tf.zeros((1,1)), dtype=tf.float32, name='bias_2')
parameter_list_2 = [w_2, b_2] # This line is changed -----------------------------------------------------------------------------
saver_2 = tf.train.Saver(parameter_list_2) # This line is changed ----------------------------------------------------------------

# model
hypothesis_2 = x_2 * w_2 + b_2

# find optimal parameter
with tf.name_scope('Train_2') as Unit_2:
    cost_2 = tf.reduce_mean(tf.square(hypothesis_2 - y_2))
    train_2 = tf.train.GradientDescentOptimizer(LEARNING_RATE_2).minimize(cost_2)

with tf.Session() as Recall_Linear_regression:
    saver = tf.train.Saver( {"weight": w_2, "bias": b_2} )
    saver.restore(Recall_Linear_regression, SAVE_PATH)

    # predict output for a new input data
    x_data_new = [[0.]]
    y_data_predicted = Recall_Linear_regression.run(hypothesis_2, feed_dict={x_2: x_data_new})
    print 'New input : ', x_data_new
    print 'Predicted output : ', y_data_predicted

    x_data_new = [[7.], [-3], [-2.]]
    y_data_predicted = Recall_Linear_regression.run(hypothesis_2, feed_dict={x_2: x_data_new})
    print 'New input : ', x_data_new
    print 'Predicted output : ', y_data_predicted

    x_data_new = [[7.]]
    y_data_predicted, w_2_now, b_2_now = Recall_Linear_regression.run([hypothesis_2, w_2, b_2], feed_dict={x_2: x_data_new})
    print 'New input : ', x_data_new
    print 'Predicted output : ', y_data_predicted
    print 'w_2 : ', w_2_now
    print 'b_2 : ', b_2_now

    for step_index in range(EPOCH_2):
        if step_index % DISPLAY_STEP_2 != 0:
            Recall_Linear_regression.run(train_2, feed_dict=feed_info_2)
        if step_index % DISPLAY_STEP_2 == 0:
            _, cost_2_now, w_2_now, b_2_now = Recall_Linear_regression.run([train_2, cost_2, w_2, b_2], feed_dict=feed_info_2)
            writer.add_summary(merged_now, step_index)
            print '========================================='
            print 'Step : ', step_index
            print 'Cost : ', cost_2_now
            print 'Weight : ', w_2_now
            print 'Bias : ', b_2_now

    saver.save(Recall_Linear_regression, SAVE_PATH_2)  # This line is changed ----------------------------------------------------

# cd Dropbox/Tensorflow/13\ Linear regression/2\ Linear regression/temp/Linear regression logs/
# tensorboard --logdir=./
# http://localhost:6006/





