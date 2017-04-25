import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create Initial variable for weight value.
def init_variable(shape):
    return(tf.Variable(tf.truncated_normal(shape = shape)))

# Fix random variable
seed = 0
tf.set_random_seed(seed)
np.random.seed(seed)

# Import data
DF = pd.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
    header=None, sep=',')
Features = DF.ix[:, 0:3].values
ClassStr = DF.ix[:, 4].values
df = np.zeros((len(ClassStr), 7))
for i in range(len(ClassStr)):              # Transform String to Number.
    df[i, 0:4] = Features[i, :]
    if ClassStr[i] == 'Iris-setosa':
        df[i, 4] = 1                        # [1, 0, 0]
    elif ClassStr[i] == 'Iris-versicolor':
        df[i, 5] = 1                        # [0, 1, 0]
    elif ClassStr[i] == 'Iris-virginica':
        df[i, 6] = 1                        # [0, 0, 1]
np.random.shuffle(df)                       # Shuffle row of matrix for test.

def Problem_Plot():
    DF.columns = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Class']
    ProblemFig = plt.figure()
    plt.subplot(2, 3, 1)
    for key, val in DF.groupby('Class'):
        plt.plot(val['Sepal Length'], val['Sepal Width'], 'o', label=key)
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')

    plt.subplot(2, 3, 2)
    for key,val in DF.groupby('Class'):
        plt.plot(val['Petal Length'], val['Petal Width'], 'o', label=key)
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')

    plt.subplot(2, 3, 3)
    for key,val in DF.groupby('Class'):
        plt.plot(val['Sepal Length'], val['Petal Width'], 'o', label=key)
    plt.xlabel('Sepal Length')
    plt.ylabel('Petal Width')

    plt.subplot(2, 3, 4)
    for key,val in DF.groupby('Class'):
        plt.plot(val['Sepal Length'], val['Petal Length'], 'o', label=key)
    plt.xlabel('Sepal Length')
    plt.ylabel('Petal Length')

    plt.subplot(2, 3, 5)
    for key,val in DF.groupby('Class'):
        plt.plot(val['Sepal Width'], val['Petal Width'], 'o', label=key)
    plt.xlabel('Sepal Width')
    plt.ylabel('Petal Width')

    plt.subplot(2, 3, 6)
    for key,val in DF.groupby('Class'):
        plt.plot(val['Sepal Width'], val['Petal Length'], 'o', label=key)
    plt.xlabel('Sepal Width')
    plt.ylabel('Petal Length')
    plt.legend(loc='best', prop={'size':6})
    plt.tight_layout()
    ProblemFig.savefig('Result_IRIS_Problem.png',dpi=100)
    return
Problem_Plot()

# Training Set 70% / Test Set 30%
# 150 * 0.7 = 105 / 150 * 0.3 = 45
Training_Features = df[0:105, 0:4]  # Training Set(Features).
Training_Classes = df[0:105, 4:7]   # Training Set(Classes).
Test_Features = df[105:, 0:4]       # Test Set(Features).
Test_Classes = df[105:, 4:7]        # Test Set(Classes).

# Control panel
LEARNING_RATE = 0.01
TRAINING_EPOCHS = 1000
BATCH_SIZE = 100

# Placeholder variables
x = tf.placeholder(tf.float32)  # Input variable of Features
y = tf.placeholder(tf.float32)  # Input variable of Class

NF = 4                      # the Number of Features.
NC = 3                      # the Number of Class.
NN = 100                    # the Number of Neuron(Node).
NUM_LAYER = 5

# First Layer
#w0 = init_variable(shape=[NF, NN])/np.sqrt(NF)      # Xavier's Initialization
w0 = init_variable(shape=[NF, NN])/np.sqrt(NF/2)    # He's Initialization
b0 = init_variable(shape=[NN])
layer = tf.nn.relu(tf.matmul(x, w0) + b0)
# layer = tf.nn.sigmoid(tf.matmul(x, w0) + b0)

for Iter in range(NUM_LAYER):
    #w = init_variable(shape=[NN, NN])/np.sqrt(NN)   # Xavier's Initialization
    w = init_variable(shape=[NN, NN])/np.sqrt(NN/2) # He's Initialization
    b = init_variable(shape=[NN])
    layer = tf.nn.relu(tf.matmul(layer, w) + b)
    # layer = tf.nn.sigmoid(tf.matmul(layer, w) + b)

# Final Layer
#w = init_variable(shape=[NN, NC])/np.sqrt(NN)       # Xavier's Initialization
w = init_variable(shape=[NN, NC])/np.sqrt(NN/2)     # He's Initialization
b = init_variable(shape=[NC])
score = tf.matmul(layer, w) + b

# Optimization method
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=score))
train = tf.train.AdagradOptimizer(LEARNING_RATE).minimize(cost)
#train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)
hypothesis = tf.nn.softmax(score)

# Performance measures
prediction = tf.argmax(score, 1)
correct_prediction = tf.equal(prediction, tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Create TensorFlow session
with tf.Session() as sess:
    # Initialize variables
    tf.global_variables_initializer().run()

    # train

    cost_history = []
    for epoch in range(TRAINING_EPOCHS):
        _, cost_this_batch, hypo = sess.run([train, cost, hypothesis],
                                            feed_dict={x: Training_Features, y: Training_Classes})
        cost_history = np.append(cost_history, cost_this_batch)

    # Plot
    CostFig = plt.figure()
    plt.plot(range(len(cost_history)), cost_history)
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.axis([0, TRAINING_EPOCHS, 0, np.max(cost_history)])
    CostFig.savefig('Result_Cost.png', dpi=100)

    print("Optimization Finished!")
    print("Accuracy: ", sess.run(accuracy, feed_dict={x: Test_Features, y: Test_Classes}))
