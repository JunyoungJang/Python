# lec12: NN의 꽃 RNN 이야기 https://www.youtube.com/watch?v=-SHPG_KMUkQ&t=23s
# lab12-1: RNN - Basics https://www.youtube.com/watch?v=B5GtZuUvujQ
# lab12-2: RNN - Hi Hello Training https://www.youtube.com/watch?v=39_P23TqUnw
# lab12-3: Long Sequence RNN https://www.youtube.com/watch?v=2R6nfCNNz1U
# lab12-4: Stacked RNN + Softmax Layer https://www.youtube.com/watch?v=vwjt1ZE5-K4
# lab12-5: Dynamic RNN https://www.youtube.com/watch?v=aArdoSpdMEc
# lab12-6: RNN with Time Series Data https://www.youtube.com/watch?v=odMGK7pwTqY

# How to build a Recurrent Neural Network in TensorFlow (1/7) https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767
# Using the RNN API in TensorFlow (2/7) https://medium.com/@erikhallstrm/tensorflow-rnn-api-2bb31821b185
# Using the LSTM API in TensorFlow (3/7) https://medium.com/@erikhallstrm/using-the-tensorflow-lstm-api-3-7-5f2b97ca6b73
# Using the Multilayered LSTM API in TensorFlow (4/7) https://medium.com/@erikhallstrm/using-the-tensorflow-multilayered-lstm-api-f6e7da7bbe40
# Using the DynamicRNN API in TensorFlow (5/7) https://medium.com/@erikhallstrm/using-the-dynamicrnn-api-in-tensorflow-7237aba7f7ea
# Using the Dropout API in TensorFlow (6/7) https://medium.com/@erikhallstrm/using-the-dropout-api-in-tensorflow-2b2e6561dfeb


import tensorflow as tf, numpy as np
tf.set_random_seed(777)


num_classes = 5
input_dim = 5  # one-hot size
sequence_length = 6  # |ihello| == 6
batch_size = 1   # one sentence
hidden_size = 5  # output from the LSTM. 5 to directly predict one-hot
learning_rate = 0.1


# data
idx2char = ['h', 'i', 'e', 'l', 'o']
x_data = [[0, 1, 0, 2, 3, 3]]    # hihell
x_one_hot = [[[1, 0, 0, 0, 0],   # h 0
              [0, 1, 0, 0, 0],   # i 1
              [1, 0, 0, 0, 0],   # h 0
              [0, 0, 1, 0, 0],   # e 2
              [0, 0, 0, 1, 0],   # l 3
              [0, 0, 0, 1, 0]]]  # l 3
y_data = [[1, 0, 2, 3, 3, 4]]    # ihello


# place holder
X = tf.placeholder(tf.float32, [None, sequence_length, input_dim])  # X one-hot
Y = tf.placeholder(tf.int32, [None, sequence_length])  # Y label

# build an RNN network
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, X, initial_state=initial_state, dtype=tf.float32)

# FC layer
X_for_fc = tf.reshape(outputs, [-1, hidden_size])
# fc_w = tf.get_variable("fc_w", [hidden_size, num_classes])
# fc_b = tf.get_variable("fc_b", [num_classes])
# outputs = tf.matmul(X_for_fc, fc_w) + fc_b
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
        l, _ = sess.run([loss, train], feed_dict={X: x_one_hot, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_one_hot})
        print(i, "loss:", l, "prediction: ", result, "true Y: ", y_data)

        # print char using dic
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("\tPrediction str: ", ''.join(result_str))