import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from scipy.misc import imresize
import os


def VAE(data):
    learning_rate = 0.0001
    num_epoch = 30000
    display_step = 1000
    num_input = data.shape[1]
    X = tf.placeholder(tf.float32, [None, num_input])

    def EncoderFtn(x):
        n_hidden = 500
        n_output = 50
        keep_prob = 0.9
        w0 = tf.contrib.layers.variance_scaling_initializer()
        b0 = tf.constant_initializer(0.)

        w1 = tf.get_variable('E_Layer1', [x.get_shape()[1], n_hidden], initializer=w0)
        b1 = tf.get_variable('E_bias1', [n_hidden], initializer=b0)
        h1 = tf.matmul(x, w1) + b1
        h1 = tf.nn.sigmoid(h1)

        w2 = tf.get_variable('E_Layer2', [h1.get_shape()[1], n_hidden], initializer=w0)
        b2 = tf.get_variable('E_bias2', [n_hidden], initializer=b0)
        h2 = tf.matmul(h1, w2) + b2
        h2 = tf.nn.sigmoid(h2)
        h2 = tf.nn.dropout(h2, keep_prob)

        OutputLayer = tf.get_variable('E_Output_Layer', [h2.get_shape()[1], n_output * 2], initializer=w0)
        bias = tf.get_variable('E_Output_bias', [n_output * 2], initializer=b0)
        gaussian_params = tf.matmul(h2, OutputLayer) + bias
        mean = gaussian_params[:, :n_output]
        stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, n_output:])
        return mean, stddev

    def DecoderFtn(x):
        n_hidden = 500
        n_output = 64
        keep_prob = 1.0
        w0 = tf.contrib.layers.variance_scaling_initializer()
        b0 = tf.constant_initializer(0.)

        w1 = tf.get_variable('D_Layer1', [x.get_shape()[1], n_hidden], initializer=w0)
        b1 = tf.get_variable('D_bias1', [n_hidden], initializer=b0)
        h1 = tf.matmul(x, w1) + b1
        h1 = tf.nn.sigmoid(h1)
        h1 = tf.nn.dropout(h1, keep_prob)

        w2 = tf.get_variable('D_Layer2', [h1.get_shape()[1], n_hidden], initializer=w0)
        b2 = tf.get_variable('D_bias2', [n_hidden], initializer=b0)
        h2 = tf.matmul(h1, w2) + b2
        h2 = tf.nn.sigmoid(h2)
        h2 = tf.nn.dropout(h2, keep_prob)

        w3 = tf.get_variable('D_Layer3', [h2.get_shape()[1], n_hidden], initializer=w0)
        b3 = tf.get_variable('D_bias3', [n_hidden], initializer=b0)
        h3 = tf.matmul(h2, w3) + b3
        h3 = tf.nn.sigmoid(h3)
        h3 = tf.nn.dropout(h3, keep_prob)

        wo = tf.get_variable('D_Output_Layer', [h3.get_shape()[1], n_output], initializer=w0)
        bo = tf.get_variable('D_Output_bias', [n_output], initializer=b0)
        y = tf.sigmoid(tf.matmul(h3, wo) + bo)
        return y

    mu, sigma = EncoderFtn(X)
    encode_x = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
    decode_x = DecoderFtn(encode_x)
    X_hat = tf.clip_by_value(decode_x, 1e-8, 1 - 1e-8)

    y_true = X
    y_pred = X_hat

    loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        if os.path.exists("vae_model.meta"):
            print("Load Model")
            _ = tf.train.import_meta_graph('./vae_model.meta')
            saver.restore(sess, tf.train.latest_checkpoint('./'))
            recon = sess.run(y_pred, feed_dict={X: data})
        else:
            print('Create Model')
            sess.run(tf.global_variables_initializer())
            for epoch in range(1, num_epoch + 1):
                _, l = sess.run([optimizer, loss], feed_dict={X: data})
                if epoch % display_step == 0 or epoch == 1:
                    print('Epoch %i, Loss: %f' % (epoch, l))
            recon = sess.run(y_pred, feed_dict={X: data})
            saver.save(sess, "./vae_model")
    return recon

digits = load_digits()
data = digits.data
img_shape = (8, 8)
n_sample, n_feature = data.shape
new_data = []
for i in range(n_sample):
    new_data0 = imresize(data[i, :].reshape(img_shape), img_shape)
    new_data.append(new_data0.reshape(1, 64))

new_data = np.array(new_data)
new_data = new_data.reshape((n_sample, 64))
new_data = new_data/np.float32(256)

recon = VAE(new_data)

plt.figure(figsize=(8, 12))
for i in range(5):
    plt.subplot(5, 2, 2*i + 1)
    plt.imshow(new_data[i].reshape(8, 8), vmin=0, vmax=1, cmap="gray")
    plt.title("Test input")
    plt.colorbar()
    plt.subplot(5, 2, 2*i + 2)
    plt.imshow(recon[i].reshape(8, 8), vmin=0, vmax=1, cmap="gray")
    plt.title("Reconstruction")
    plt.colorbar()
plt.tight_layout()
plt.show()