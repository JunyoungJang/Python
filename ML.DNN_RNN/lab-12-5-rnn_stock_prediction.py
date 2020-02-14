# lec12: NN의 꽃 RNN 이야기 https://www.youtube.com/watch?v=-SHPG_KMUkQ&t=23s
# lab12-1: RNN - Basics https://www.youtube.com/watch?v=B5GtZuUvujQ
# lab12-2: RNN - Hi Hello Training https://www.youtube.com/watch?v=39_P23TqUnw
# lab12-3: Long Sequence RNN https://www.youtube.com/watch?v=2R6nfCNNz1U
# lab12-4: Stacked RNN + Softmax Layer https://www.youtube.com/watch?v=vwjt1ZE5-K4
# lab12-5: Dynamic RNN https://www.youtube.com/watch?v=aArdoSpdMEc
# lab12-6: RNN with Time Series Data https://www.youtube.com/watch?v=odMGK7pwTqY


# full batch - len(trainX) is not equal to len(testX)
import tensorflow as tf, numpy as np, matplotlib.pyplot as plt
import matplotlib, os
tf.set_random_seed(777)


if "DISPLAY" not in os.environ:
    # remove Travis CI Error
    matplotlib.use('Agg')


def MinMaxScaler(data):
    # Min Max Normalization http://sebastianraschka.com/Articles/2014_about_feature_scaling.html
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)


# hyper parameters
seq_length = 7
data_dim = 5
hidden_dim = 10
output_dim = 1
learning_rate = 0.01
iterations = 500


# build a dataset
xy = np.loadtxt('data-02-stock_daily.csv', delimiter=',')
xy = xy[::-1]  # reverse order (chronically ordered)
xy = MinMaxScaler(xy)
x = xy
y = xy[:, [-1]]  # Close as label

dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i:i + seq_length]
    _y = y[i + seq_length]  # Next close price
    print(_x, "->", _y)
    dataX.append(_x)
    dataY.append(_y)

train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size
trainX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), np.array(dataY[train_size:len(dataY)])


# place holder
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, 1])

# build an RNN network
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], output_dim, activation_fn=None)  # We use the last cell's output

loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# RMSE
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Training step
    for i in range(iterations):
        _, step_loss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
        print("[step: {}] loss: {}".format(i, step_loss))

    # Test step
    test_predict = sess.run(Y_pred, feed_dict={X: testX})
    rmse_val = sess.run(rmse, feed_dict={targets: testY, predictions: test_predict})
    print("RMSE: {}".format(rmse_val))

    # Plot predictions
    plt.plot(testY) # blue
    plt.plot(test_predict) # yellow
    plt.xlabel("Time Period")
    plt.ylabel("Stock Price")
    plt.show()

    # Plot predictions
    y1 = test_predict[1:] - test_predict[0:-1]
    y2 = testY[1:] - testY[0:-1]
    plt.plot(y1) # blue
    plt.plot(y2) # yellow
    plt.xlabel("Time Period")
    plt.ylabel("Stock Price Change")
    plt.show()