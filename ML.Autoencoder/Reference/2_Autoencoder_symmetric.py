import tensorflow as tf, numpy as np, matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
# mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
mnist = input_data.read_data_sets("/Users/sungchul/Dropbox/Data/MNIST/", one_hot=True)


learning_rate = 0.01
training_epochs = 20
batch_size = 256
display_step = 1
examples_to_show = 10

# Network Parameters
n_hidden_1 = 256  # 1st layer num features
n_hidden_2 = 128  # 2nd layer num features
n_input = 784  # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

with tf.variable_scope('encoder'):
    eh1 = tf.get_variable(name='eh1', initializer=tf.random_normal([n_input, n_hidden_1]))
    eh2 = tf.get_variable(name='ehe', initializer=tf.random_normal([n_hidden_1, n_hidden_2]))

with tf.variable_scope('encoder', reuse=True):
    eh1c = tf.get_variable(name='eh1')
    eh2c = tf.get_variable(name='ehe')

with tf.variable_scope('decoder'):
    dh1 = tf.transpose(eh2c)
    dh2 = tf.transpose(eh1c)

biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}


def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, eh1), biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, eh2), biases['encoder_b2']))
    return layer_2


def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, dh1), biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, dh2), biases['decoder_b2']))
    return layer_2

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)


# Launch the graph
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    total_batch = int(mnist.train.num_examples/batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")

    # Applying encode and decode over test set
    encode_decode = sess.run(
        y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
    # Compare original images with their reconstructions
    plt.figure(1)
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    f.show()
    plt.draw()
    plt.waitforbuttonpress()