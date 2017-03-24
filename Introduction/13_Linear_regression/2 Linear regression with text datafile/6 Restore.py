import tensorflow as tf
import numpy as np
import pandas as pd

LEARNING_RATE = 0.1
EPOCH = 100
DISPLAY_STEP = 2
LOG_DIR = "./temp/logfile"
SAVE_PATH = LOG_DIR + "/savefile.ckpt"

INPUT_SIZE = 3
OUTPUT_SIZE = 3

df = pd.read_csv('train_data_linear_regression.txt', sep='\s+')
df = np.array(df, np.float32)
x_data = df[:, 0:-1]
y_data = np.expand_dims( df[:, -1], axis=1)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform((INPUT_SIZE, OUTPUT_SIZE), -1., 1.), dtype=tf.float32, name='weight')
parameter_list = [W]
saver = tf.train.Saver(parameter_list)

with tf.name_scope('Hypothesis') as Unit_1:
    hypothesis = tf.matmul(X, W)

with tf.name_scope('Train') as Unit_2:
    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    train = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

W_summ = tf.summary.histogram('weights_Record', W)
cost_summ = tf.summary.scalar('cost_Record', cost)
merged = tf.summary.merge_all()

with tf.Session() as sess:
    writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
    tf.global_variables_initializer().run()

    for step in range(EPOCH):
        if step % DISPLAY_STEP != 0:
            sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % DISPLAY_STEP == 0:
            _, cost_now, W_now, merged_now = sess.run([train, cost, W, merged], feed_dict={X: x_data, Y: y_data})
            writer.add_summary(merged_now, step)
            print step, cost_now, W_now

    saver.save(sess, SAVE_PATH)

print "======================================================== Restoration ==========================================="

W_2 = tf.Variable(tf.random_uniform((INPUT_SIZE, OUTPUT_SIZE), -1., 1.), tf.float32, name='Weights_2') # This line is changed ----------

with tf.Session() as sess_2:  # This line is changed -----------------------------------------------------------------------------
        saver = tf.train.Saver({"weight": W_2})  # This line is changed ----------------------------------------------------------
        saver.restore(sess_2, SAVE_PATH)  # This line is changed -----------------------------------------------------------------

# cd Dropbox/Tensorflow/13\ Linear\ regression/3\ Linear\ regression\ with\ text\ datafile/temp/Linear\ regression\ with\ text\ datafile\ logs/
# tensorboard --logdir=./
# http://localhost:6006/


