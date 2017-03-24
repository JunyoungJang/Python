import pandas as pd
import numpy as np
import tensorflow as tf

LEARNING_RATE = 0.001
LOG_DIR = "./temp/logfile"
INPUT_SIZE = 1
OUTPUT_SIZE = 29
EPOCH_TRAIN = 30000
DISPLAY_STEP = 100
BATCH_SIZE = 100

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

W = tf.get_variable(shape=(INPUT_SIZE, OUTPUT_SIZE), initializer=tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32), name='weight')
B = tf.Variable(tf.zeros((1, OUTPUT_SIZE)), dtype=tf.float32, name='bias')

with tf.name_scope('Hypothesis') as Unit_1:
    hypothesis = tf.matmul(X, W) + B

with tf.name_scope('Train') as Unit_2:
    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

W_summ = tf.summary.histogram('weights_Record', W)
B_summ = tf.summary.histogram('bias_Record', B)
cost_summ = tf.summary.scalar('cost_Record', cost)
merged = tf.summary.merge_all()
cost_record = []

def Data_Shuffle(X_DATA, Y_DATA):
    number_of_samples = len(X_DATA)
    a = np.arange(number_of_samples)
    np.random.shuffle(a)
    X_DATA_shuffled = X_DATA[a,:]
    Y_DATA_shuffled = Y_DATA[a, :]
    return X_DATA_shuffled, Y_DATA_shuffled

with tf.Session() as sess:
    writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
    tf.global_variables_initializer().run()

    for epoch in range(EPOCH_TRAIN):
        print epoch
        x_data, y_data = Data_Shuffle(x_data, y_data)
        number_of_batches = int( len(x_data) / BATCH_SIZE )
        for batch_number in range(number_of_batches):
            x_data_batch = x_data[batch_number*BATCH_SIZE:(batch_number+1)*BATCH_SIZE, :]
            y_data_batch = y_data[batch_number * BATCH_SIZE:(batch_number + 1) * BATCH_SIZE, :]
            if (batch_number+1) % number_of_batches != 0:
                sess.run(train, feed_dict={X: x_data_batch, Y: y_data_batch})
            if (batch_number+1) % number_of_batches == 0:
                _, merged_now = sess.run([train, merged], feed_dict={X: x_data, Y: y_data})
                writer.add_summary(merged_now, epoch)
