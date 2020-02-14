# lec12: NN의 꽃 RNN 이야기 https://www.youtube.com/watch?v=-SHPG_KMUkQ&t=23s
# lab12-1: RNN - Basics https://www.youtube.com/watch?v=B5GtZuUvujQ
# lab12-2: RNN - Hi Hello Training https://www.youtube.com/watch?v=39_P23TqUnw
# lab12-3: Long Sequence RNN https://www.youtube.com/watch?v=2R6nfCNNz1U
# lab12-4: Stacked RNN + Softmax Layer https://www.youtube.com/watch?v=vwjt1ZE5-K4
# lab12-5: Dynamic RNN https://www.youtube.com/watch?v=aArdoSpdMEc
# lab12-6: RNN with Time Series Data https://www.youtube.com/watch?v=odMGK7pwTqY


# full batch
import tensorflow as tf, numpy as np
from tensorflow.contrib import rnn
tf.set_random_seed(777)


# build a dictionary of index -> char and char -> idex
sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")
char_set = list(set(sentence))
char_dic = {w: i for i, w in enumerate(char_set)}


# hyper parameters
data_dim = len(char_set)
hidden_size = len(char_set)
num_classes = len(char_set)
sequence_length = 10  # Any arbitrary number
learning_rate = 0.1


# build a data set
dataX = []
dataY = []
for i in range(0, len(sentence) - sequence_length):
    x_str = sentence[i:i + sequence_length]
    y_str = sentence[i + 1: i + sequence_length + 1]
    print(i, x_str, '->', y_str)
    x = [char_dic[c] for c in x_str]  # x str to index
    y = [char_dic[c] for c in y_str]  # y str to index
    dataX.append(x)
    dataY.append(y)

batch_size = len(dataX) # hyper parameters


# place holder
X = tf.placeholder(tf.int32, [None, sequence_length]) # input - not one hot, should be transformed to one hot
Y = tf.placeholder(tf.int32, [None, sequence_length])

# One-hot encoding
X_one_hot = tf.one_hot(X, num_classes) # [None, sequence_length] -> [None, sequence_length, num_classes]
print(X_one_hot)  # check out the shape


# Make a lstm cell with hidden_size (each unit output vector size)
# def lstm_cell():
#     cell = rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
#     return cell
# cells = rnn.MultiRNNCell([lstm_cell() for _ in range(2)], state_is_tuple=True)
cell_1 = rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
cell_2 = rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
cells = rnn.MultiRNNCell([cell_1, cell_2], state_is_tuple=True)
initial_state = cells.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cells, X_one_hot, initial_state=initial_state, dtype=tf.float32)

# FC layer
X_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn=None)
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(500):
        _, l, results = sess.run([train_op, loss, outputs], feed_dict={X: dataX, Y: dataY})
        for j, result in enumerate(results):
            index = np.argmax(result, axis=1)
            print(i, j, ''.join([char_set[t] for t in index]), l)

    # Let's print the last char of each result to check it works
    results = sess.run(outputs, feed_dict={X: dataX})
    for j, result in enumerate(results):
        index = np.argmax(result, axis=1)
        if j is 0:  # print all for the first result to make a sentence
            print(''.join([char_set[t] for t in index]), end='')
        else:
            print(char_set[index[-1]], end='')