import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imsave
from tensorflow.examples.tutorials.mnist import input_data
import os


# Define Global Parameters
n_latent = 2
n_gen = 100
learning_rate = 0.00001
num_epoch = 100000
display_step = 1000
batch_size = 100
n_hidden = 500

# VAE Util
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1.0/tf.sqrt(in_dim/2.0)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def Encoder(x):
    global n_hidden
    keep_prob = 0.9

    # Hidden Layer1 of Encoder
    w1 = tf.Variable(xavier_init([x.get_shape()[1].value, n_hidden]))
    b1 = tf.Variable(tf.zeros(shape=[n_hidden]))
    h1 = tf.matmul(x, w1) + b1
    h1 = tf.nn.sigmoid(h1)

    # Hidden Layer2 of Encoder
    w2 = tf.Variable(xavier_init([h1.get_shape()[1].value, n_hidden]))
    b2 = tf.Variable(tf.zeros(shape=[n_hidden]))
    h2 = tf.matmul(h1, w2) + b2
    h2 = tf.nn.sigmoid(h2)
    h2 = tf.nn.dropout(h2, keep_prob)
    return h2

def Decoder(x):
    global n_hidden
    n_output = 784  # 28 by 28 Image of Tensor MNIST
    keep_prob = 1.0

    # Hidden Layer1 of Decoder
    w1 = tf.Variable(xavier_init([x.get_shape()[1].value, n_hidden]))
    b1 = tf.Variable(tf.zeros(shape=[n_hidden]))
    h1 = tf.matmul(x, w1) + b1
    h1 = tf.nn.sigmoid(h1)
    h1 = tf.nn.dropout(h1, keep_prob)

    # Hidden Layer2 of Decoder
    w2 = tf.Variable(xavier_init([h1.get_shape()[1].value, n_hidden]))
    b2 = tf.Variable(tf.zeros(shape=[n_hidden]))
    h2 = tf.matmul(h1, w2) + b2
    h2 = tf.nn.sigmoid(h2)
    h2 = tf.nn.dropout(h2, keep_prob)

    # Hidden Layer3 of Decoder
    w3 = tf.Variable(xavier_init([h2.get_shape()[1].value, n_hidden]))
    b3 = tf.Variable(tf.zeros(shape=[n_hidden]))
    h3 = tf.matmul(h2, w3) + b3
    h3 = tf.nn.sigmoid(h3)
    h3 = tf.nn.dropout(h3, keep_prob)

    # Output Layer of Decoder
    wo = tf.Variable(xavier_init([h3.get_shape()[1].value, n_output]))
    bo = tf.Variable(tf.zeros(shape=[n_output]))
    y = tf.sigmoid(tf.matmul(h3, wo) + bo)
    return y

# DCGAN Util
def Sample_Z(m, n):
    return np.random.uniform(0., 1., size=[m, n])

def Conv2d(input, output_dim=64, kernel=(5, 5), strides=(2, 2), stddev=0.2, name='Conv_2d'):
    with tf.variable_scope(name):
        W = tf.get_variable('Conv2dW', [kernel[0], kernel[1], input.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable('Conv2db', [output_dim], initializer=tf.zeros_initializer())
        return tf.nn.conv2d(input, W, strides=[1, strides[0], strides[1], 1], padding='SAME') + b

def Deconv2d(input, output_dim, batch_size, kernel=(5, 5), strides=(2, 2), stddev=0.2, name='DeConv_2d'):
    with tf.variable_scope(name):
        W = tf.get_variable('Deconv2dW', [kernel[0], kernel[1], output_dim, input.get_shape()[-1]],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable('Deconv2db', [output_dim], initializer=tf.zeros_initializer())
        input_shape = input.get_shape().as_list()
        output_shape = [batch_size, int(input_shape[1] * strides[0]), int(input_shape[2] * strides[1]), output_dim]
        deconv = tf.nn.conv2d_transpose(input, W, output_shape=output_shape, strides=[1, strides[0], strides[1], 1])
        return deconv + b

def Dense(input, output_dim, stddev=0.02, name='Dense'):
    with tf.variable_scope(name):
        shape = input.get_shape()
        W = tf.get_variable('Weight', [shape[1], output_dim], tf.float32, tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable('bias', [output_dim], initializer=tf.zeros_initializer())
        return tf.matmul(input, W) + b

def BatchNormalization(input, name='BatchNormal'):
    with tf.variable_scope(name):
        output_dim = input.get_shape()[-1]
        beta = tf.get_variable('BnBeta', [output_dim], initializer=tf.zeros_initializer())
        gamma = tf.get_variable('BnGamma', [output_dim], initializer=tf.ones_initializer())
        if len(input.get_shape()) == 2:
            mean, var = tf.nn.moments(input, [0])
        else:
            mean, var = tf.nn.moments(input, [0, 1, 2])
        return tf.nn.batch_normalization(input, mean, var, beta, gamma, 1e-5)

def lrelu(input, leak=0.2):
    return tf.maximum(input, leak * input)

def Generator(z, name='Generator_Operator'):
    with tf.variable_scope(name):
        # Dense(Neural) Layer.
        w1 = Dense(z, output_dim=1024, name='Generator_w1')
        b1 = BatchNormalization(w1, name='Generator_b1')
        Layer1 = tf.nn.relu(b1)
        # Dense(Neural) Layer.
        w2 = Dense(Layer1, output_dim=7*7*128, name='Generator_w2')
        b2 = BatchNormalization(w2, name='Generator_b2')
        Layer2 = tf.reshape(tf.nn.relu(b2), [-1, 7, 7, 128])
        # Convolution Layer.
        Conv1 = Deconv2d(Layer2, output_dim=64, batch_size=batch_size, name='Generator_DeConv3')
        Conv1 = BatchNormalization(Conv1, name='Generator_b3')
        Layer3 = tf.nn.relu(Conv1)
        # Output Layer.
        Conv2 = Deconv2d(Layer3, output_dim=1, batch_size=batch_size, name='Generator_DeConv4')
        Conv2 = tf.reshape(Conv2, [-1, 784])
        return tf.nn.sigmoid(Conv2)

def Discriminator(X, reuse=False, name='Discriminator_Operator'):
    with tf.variable_scope(name, reuse=reuse):
        if len(X.get_shape()) > 2:
            Conv1 = Conv2d(X, output_dim=64, name='Discriminator_Conv1')
        else:
            Conv1 = Conv2d(tf.reshape(X, [-1, 28, 28, 1]), output_dim=64, name='Discriminator_Conv1')
        Layer1 = lrelu(Conv1)
        # Convolution Layer.
        Conv2 = lrelu(Conv2d(Layer1, output_dim=128, name='Discriminator_Conv2'))
        Layer2 = tf.nn.dropout(lrelu(tf.reshape(Conv2, [-1, 256])), 0.5)
        # Output Layer.
        Layer3 = Dense(Layer2, output_dim=1, name='Discriminator_Output')
        return tf.nn.sigmoid(Layer3)


###########
## Model ##
###########
# 1. Variational AutoEncoder (VAE)
def VAE(data):
    global n_latent
    global learning_rate
    global num_epoch
    global display_step
    global batch_size
    tf.reset_default_graph()
    num_input = data.train.images.shape[1]
    X = tf.placeholder(tf.float32, [None, num_input])

    EnLayer = Encoder(X)
    BottleNeck = tf.Variable(xavier_init([EnLayer.get_shape()[1].value, n_latent * 2]))
    bias = tf.Variable(tf.zeros([n_latent * 2]))
    gaussian_params = tf.matmul(EnLayer, BottleNeck) + bias
    mu = gaussian_params[:, :n_latent]
    sigma = 1e-6 + tf.nn.softplus(gaussian_params[:, n_latent:])

    # Latent space
    latent = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)

    # Decoder
    Decode_X = Decoder(latent)
    X_hat = tf.clip_by_value(Decode_X, 1e-8, 1 - 1e-8)

    y_true = X
    y_pred = X_hat

    # Error Criterion using ELBO.
    marginal_likelihood = tf.reduce_sum(y_true * tf.log(y_pred) + (1 - y_true) * tf.log(1 - y_pred), 1)
    KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, 1)
    marginal_likelihood = tf.reduce_mean(marginal_likelihood)
    KL_divergence = tf.reduce_mean(KL_divergence)
    ELBO = marginal_likelihood - KL_divergence
    loss = -ELBO
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    saver_vae = tf.train.Saver()
    if os.path.exists("./vae/vae_model.meta"):
        with tf.Session() as sess:
            print("----------------------- Load VAE Model")
            _ = tf.train.import_meta_graph('./vae/vae_model.meta')
            saver_vae.restore(sess, tf.train.latest_checkpoint('./vae/'))

            batch_xs, _ = data.train.next_batch(100)
            recon = sess.run(y_pred, feed_dict={X: batch_xs})

            # Reconstructed mnist test data.
            nx = ny = 6
            re_canvas = np.empty((28 * ny, 28 * nx))
            for ii in range(nx):
                for j in range(ny):
                    re_canvas[(nx - ii - 1) * 28:(nx - ii) * 28, j * 28:(j + 1) * 28] = recon[nx*ii+j].reshape(28, 28)
            imsave('./re_vae.png', re_canvas)
            return recon
    else:
        with tf.Session() as sess:
            print('Create VAE Model')
            sess.run(tf.global_variables_initializer())
            total_batch = int(100 / batch_size)
            for epoch in range(1, num_epoch + 1):
                for i in range(total_batch):
                    batch_xs, _ = data.train.next_batch(batch_size)
                    _, l = sess.run([optimizer, loss], feed_dict={X: batch_xs})
                if epoch % display_step == 0 or epoch == 1:
                    print('Epoch %i, Loss: %f' % (epoch, l))
            saver_vae.save(sess, "./vae/vae_model")

# 2. Deep Convolution Generative Adversarial Nets (DCGAN)
def DCGAN(data):
    global learning_rate
    global num_epoch
    global display_step
    global batch_size
    global n_gen
    tf.reset_default_graph()
    num_input = data.train.images.shape[1]
    X = tf.placeholder(tf.float32, [None, num_input])
    Z = tf.placeholder(tf.float32, [None, n_gen])

    image_real = X
    image_gen = Generator(Z, 'Generator')
    D_real = Discriminator(image_real, False, 'Discriminator')
    D_fake = Discriminator(image_gen, True, 'Discriminator')

    D_loss = -tf.reduce_mean(tf.log(D_real) - tf.log(D_fake))
    G_loss = -tf.reduce_mean(tf.log(D_fake))

    vars = tf.trainable_variables()
    d_params = [v for v in vars if v.name.startswith('Discriminator/')]
    g_params = [v for v in vars if v.name.startswith('Generator/')]

    D_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.1).minimize(D_loss, var_list=d_params)
    G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.3).minimize(G_loss, var_list=g_params)

    saver_gan = tf.train.Saver()
    if os.path.exists("./dcgan/dcgan_model.meta"):
        with tf.Session() as sess:
            print("----------------------- Load DCGAN Model")
            sess.run(tf.global_variables_initializer())
            _ = tf.train.import_meta_graph('./dcgan/dcgan_model.meta')
            saver_gan.restore(sess, tf.train.latest_checkpoint('./dcgan/'))
            recon = sess.run(image_gen, feed_dict={Z: Sample_Z(batch_size, n_gen)})
            # Reconstructed mnist test data.
            nx = ny = 6
            re_canvas = np.empty((28 * ny, 28 * nx))
            for ii in range(nx):
                for j in range(ny):
                    re_canvas[(nx - ii - 1) * 28:(nx - ii) * 28, j * 28:(j + 1) * 28] = recon[nx * ii + j].reshape(28, 28)
            imsave('./re_dcgan.png', re_canvas)
            return recon
    else:
        with tf.Session() as sess:
            print('Create DCGAN Model')
            sess.run(tf.global_variables_initializer())
            total_batch = int(100 / batch_size)
            for epoch in range(1, num_epoch + 1):
                G_score = []
                D_score = []
                for i in range(total_batch):
                    batch_xs, _ = data.train.next_batch(batch_size)
                    _, D_loss_curr = sess.run([D_solver, D_loss],
                                              feed_dict={X: batch_xs, Z: Sample_Z(batch_xs.shape[0], n_gen)})
                    D_score.append(D_loss_curr)
                    _, G_loss_curr, recon = sess.run([G_solver, G_loss, image_gen],
                                              feed_dict={X: batch_xs, Z: Sample_Z(batch_xs.shape[0], n_gen)})
                    G_score.append(G_loss_curr)

                if epoch % display_step == 0 or epoch == 1:
                    # Reconstructed mnist test data.
                    nx = ny = 6
                    re_canvas = np.empty((28 * ny, 28 * nx))
                    for ii in range(nx):
                        for j in range(ny):
                            re_canvas[(nx - ii - 1) * 28:(nx - ii) * 28, j * 28:(j + 1) * 28] = recon[nx * ii + j].reshape(28, 28)
                    picname = 're_dcgan_%04d.png' % epoch
                    imsave(picname, re_canvas)
                    print('Epoch %i, GLoss: %e, DLoss: %e' % (epoch, np.mean(G_score), np.mean(D_score)))
            saver_gan.save(sess, "./dcgan/dcgan_model")


# If it do not exist storage, make folder to save model.
if not os.path.exists("vae"):
    os.makedirs("vae")
if not os.path.exists("dcgan"):
    os.makedirs("dcgan")

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
# VAE
if os.path.exists("./vae/vae_model.meta"):
    # Test
    recon_VAE = VAE(mnist)
else:
    # Train
    VAE(mnist)
    # Test
    recon_VAE = VAE(mnist)

# vanilla GAN
if os.path.exists("./dcgan/dcgan_model.meta"):
    # Test
    recon_DCGAN = DCGAN(mnist)
else:
    # Train
    DCGAN(mnist)
    # Test
    recon_DCGAN = DCGAN(mnist)

plt.figure(figsize=(8, 12))
for i in range(3):
    plt.subplot(3, 2, 2*i + 1)
    plt.imshow(recon_VAE[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
    plt.title("VAE")
    plt.colorbar()
    plt.subplot(3, 2, 2*i + 2)
    plt.imshow(recon_DCGAN[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
    plt.title("DCGAN")
    plt.colorbar()
plt.tight_layout()
plt.savefig('./compare_figure.png')
print('Done')