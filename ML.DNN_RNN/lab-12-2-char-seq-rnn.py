# lec12: NN의 꽃 RNN 이야기 https://www.youtube.com/watch?v=-SHPG_KMUkQ&t=23s
# lab12-1: RNN - Basics https://www.youtube.com/watch?v=B5GtZuUvujQ
# lab12-2: RNN - Hi Hello Training https://www.youtube.com/watch?v=39_P23TqUnw
# lab12-3: Long Sequence RNN https://www.youtube.com/watch?v=2R6nfCNNz1U
# lab12-4: Stacked RNN + Softmax Layer https://www.youtube.com/watch?v=vwjt1ZE5-K4
# lab12-5: Dynamic RNN https://www.youtube.com/watch?v=aArdoSpdMEc
# lab12-6: RNN with Time Series Data https://www.youtube.com/watch?v=odMGK7pwTqY


# input - not one hot, should be transformed to one hot
import tensorflow as tf, numpy as np
tf.set_random_seed(777)


# build a dictionary of index -> char and char -> idex
sample = "if you want you"
idx2char = list(set(sample))  # index -> char
char2idx = {c: i for i, c in enumerate(idx2char)}  # char -> idex


# hyper parameters
dic_size = len(char2idx)  # RNN input size (one hot size)
hidden_size = len(char2idx)  # RNN output size
num_classes = len(char2idx)  # final output size (RNN or softmax, etc.)
batch_size = 1  # one sample data, one batch
sequence_length = len(sample) - 1  # number of lstm rollings (unit #)
learning_rate = 0.1


# build a data set
sample_idx = [char2idx[c] for c in sample]  # char to index
x_data = [sample_idx[:-1]]  # X data sample (0 ~ n-1) hello: hell
y_data = [sample_idx[1:]]   # Y label sample (1 ~ n) hello: ello


# place holder
X = tf.placeholder(tf.int32, [None, sequence_length])  # input - not one hot, should be transformed to one hot
Y = tf.placeholder(tf.int32, [None, sequence_length])  # Y label

# build an RNN network
x_one_hot = tf.one_hot(X, num_classes)  # [None, sequence_length] -> [None, sequence_length, num_classes]
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=initial_state, dtype=tf.float32)

# FC layer
X_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(inputs=X_for_fc, num_outputs=num_classes, activation_fn=None)
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(50):
        l, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_data})

        # print char using dic
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print(i, "loss:", l, "Prediction:", ''.join(result_str))