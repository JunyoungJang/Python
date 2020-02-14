import tensorflow as tf
import pandas as pd
import numpy as np

# Create Initial variable
def init_variable(shape):
    return(tf.Variable(tf.truncated_normal(shape = shape)))

# Fix random variable
seed = 100
tf.set_random_seed(seed)

# Import data
DF = pd.read_excel('proto.xlsx', sheetname=0)
df = np.array(DF.iloc[1:380, 3:9])

# Control panel
LEARNING_RATE = 0.01
TRAINING_EPOCHS = 1000
BATCH_SIZE = 100

# Placeholder variables
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

NF = 3
NN = 100
NUM_LAYER = 5

# First Layer
w0 = init_variable(shape=[NF, NN])/np.sqrt(NF/2)
b0 = init_variable(shape=[NN])
layer = tf.nn.relu(tf.matmul(x, w0) + b0)

for Iter in range(NUM_LAYER):
    w = init_variable(shape=[NN, NN])/np.sqrt(NN/2)
    b = init_variable(shape=[NN])
    layer = tf.nn.relu(tf.matmul(layer, w) + b)

# Final Layer
w = init_variable(shape=[NN, NF])/np.sqrt(NN/2)
b = init_variable(shape=[NF])
score = tf.matmul(layer, w) + b

# Optimization method
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=score))
train = tf.train.AdagradOptimizer(LEARNING_RATE).minimize(cost)

# Performance measures
prediction = tf.argmax(score, 1)
correct_prediction = tf.equal(prediction, tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Create TensorFlow session
with tf.Session() as sess:
    # Initialize variables
    tf.global_variables_initializer().run()

    # train
    for epoch in range(TRAINING_EPOCHS):
        _, cost_this_batch = sess.run([train, cost], feed_dict={x: df[0:300, 0:3], y: df[0:300, 3:6]})
    print("Optimization Finished!")

    print("Accuracy:", sess.run(accuracy, feed_dict={x: df[300:, 0:3], y: df[300:, 3:6]}))
