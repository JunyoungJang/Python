import pandas as pd
import numpy as np
import tensorflow as tf

LEARNING_RATE = 0.001
LOG_DIR = "./temp/logfile"
INPUT_SIZE = 1
OUTPUT_SIZE = 29
EPOCH_TRAIN = 30000
DISPLAY_STEP = 100

df = pd.read_excel('/Users/sungchul/Dropbox/Data/DOW30.xlsx', sheetname=0)
df = df.iloc[:, range(0, df.shape[1], 2)]
df = df.dropna(axis=1)
df = np.array(df, np.float32)

def Daily_Return_Computation(Daily_Adjust_Close):
    c = Daily_Adjust_Close
    r = (c[1:,:]-c[0:-1,:])/c[0:-1,:]
    return r

retrun_data = Daily_Return_Computation(Daily_Adjust_Close=df)

x_data = np.expand_dims(retrun_data[:,-1], axis=1)
y_data = retrun_data[:,0:-1]

X = tf.placeholder(dtype=tf.float32)
Y = tf.placeholder(dtype=tf.float32)

W = tf.Variable(tf.random_uniform((INPUT_SIZE, OUTPUT_SIZE), -1., 1.), dtype=tf.float32, name='weight')
B = tf.Variable(tf.zeros((1, OUTPUT_SIZE)), dtype=tf.float32, name='bias')

with tf.name_scope('Hypothesis') as Unit_1:
    hypothesis = tf.matmul(X, W) + B

with tf.name_scope('Train') as Unit_2:
    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    train = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

W_summ = tf.summary.histogram('weights_Record', W)
B_summ = tf.summary.histogram('bias_Record', B)
cost_summ = tf.summary.scalar('cost_Record', cost)
merged = tf.summary.merge_all()
cost_record = []

with tf.Session() as sess:
    writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
    tf.global_variables_initializer().run()

    for step in range(EPOCH_TRAIN):
        if step % DISPLAY_STEP != 0:
            sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % DISPLAY_STEP == 0:
            _, cost_now, W_now, merged_now = sess.run([train, cost, W, merged], feed_dict={X: x_data, Y: y_data})
            writer.add_summary(merged_now, step)
            print cost_now, step
