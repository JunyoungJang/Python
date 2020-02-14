import tensorflow as tf, numpy as np, matplotlib.pyplot as plt
import random
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


class NeuralNet:
    def __init__(self, enc, act_ft):
        """
        :param enc: list, encoder
        :param act_ft: string, activated function
        """
        self.enc = enc  # list
        self.act_ft = act_ft

        if self.act_ft is 'sig':
            self.act = tf.nn.sigmoid
            self.b_init = 0.00
        elif self.act_ft is 'tanh':
            self.act = tf.nn.tanh
            self.b_init = 0.00
        elif self.act_ft is 'relu':
            self.act = tf.nn.relu
            self.b_init = 0.01
        elif self.act_ft is 'elu':
            self.act = tf.nn.elu
            self.b_init = 0.00

        # placeholder
        self.input_holder = tf.placeholder(dtype=tf.float32, name='input')
        self.target = tf.placeholder(dtype=tf.float32, name='target')
        self.output = [self.input_holder]


        # train
        self.logits = self.inference()
        self.opt = tf.train.AdamOptimizer()  # default 0.001
        self.train_op = self.opt.minimize(self.loss())

    def weight_var(self, n_in, n_out, layer_name):
        weight = tf.get_variable(name='W'+str(layer_name), shape=[n_in, n_out],
                                 initializer=tf.contrib.layers.xavier_initializer())
        tf.summary.histogram('weight', weight)
        return weight

    def bias_var(self, n_out, b_init, layer_name):
        bias = tf.get_variable(name='b'+str(layer_name), initializer=tf.constant(b_init, shape=[n_out]))
        tf.summary.histogram('bias', bias)
        return bias

    def inference(self):

        with tf.variable_scope('encoder'):
            W1 = self.weight_var(self.enc[0], self.enc[1], 1)
            b1 = self.bias_var(self.enc[1], self.b_init, 1)
            output = self.act(tf.matmul(self.output[-1], W1) + b1)
            self.output.append(output)

            W2 = self.weight_var(self.enc[1], self.enc[2], 2)
            b2 = self.bias_var(self.enc[2], self.b_init, 2)
            output = self.act(tf.matmul(self.output[-1], W2) + b2)
            self.output.append(output)

            W3 = self.weight_var(self.enc[2], self.enc[3], 3)
            b3 = self.bias_var(self.enc[3], self.b_init, 3)
            output = self.act(tf.matmul(self.output[-1], W3) + b3)
            self.output.append(output)

            # W4 = self.weight_var(self.enc[3], self.enc[4], 4)
            # b4 = self.bias_var(self.enc[4], self.b_init, 4)
            # output = self.act(tf.matmul(self.output[-1], W4) + b4)
            # self.output.append(output)

        with tf.variable_scope('encoder', reuse=True):
            W1c = tf.get_variable(name='W'+str(1), shape=[self.enc[0], self.enc[1]])
            W2c = tf.get_variable(name='W'+str(2), shape=[self.enc[1], self.enc[2]])
            W3c = tf.get_variable(name='W'+str(3), shape=[self.enc[2], self.enc[3]])
            # W4c = tf.get_variable(name='W'+str(4), shape=[self.enc[3], self.enc[4]])

        with tf.variable_scope('decoder'):
            # W5 = tf.transpose(W4c)
            # b5 = self.bias_var(self.enc[3], self.b_init, 5)
            # output = self.act(tf.matmul(self.output[-1], W5) + b5)
            # self.output.append(output)

            W6 = tf.transpose(W3c)
            b6 = self.bias_var(self.enc[2], self.b_init, 6)
            output = self.act(tf.matmul(self.output[-1], W6) + b6)
            self.output.append(output)

            W7 = tf.transpose(W2c)
            b7 = self.bias_var(self.enc[1], self.b_init, 7)
            output = self.act(tf.matmul(self.output[-1], W7) + b7)
            self.output.append(output)

            W8 = tf.transpose(W1c)
            b8 = self.bias_var(self.enc[0], 0.00, 8)
            output = self.act(tf.matmul(self.output[-1], W8) + b8)
            self.output.append(output)
            return self.output[-1]

    def loss(self):
        loss = tf.reduce_mean(tf.square(self.logits - self.target), name='loss')
        tf.summary.scalar('loss', loss)
        return loss


if __name__ == '__main__':

    # parameter
    num_train = 50000
    batch_size = 400

    x = np.arange(-1, 1, 0.05)
    y = np.arange(-1, 1, 0.05)
    x_grid, y_grid = np.meshgrid(x, y)
    z0_grid = x_grid**2 + y_grid**2

    np.random.seed(1)
    e = np.random.randn(x_grid.shape[0], x_grid.shape[1])
    z_grid = z0_grid + 0.3*e

    X = x_grid.reshape([-1, 1])
    Y = y_grid.reshape([-1, 1])
    Z = z_grid.reshape([-1, 1])

    samples = np.hstack([X, Y, Z])

    with tf.Session() as sess:
        nnet = NeuralNet([3, 10, 10, 2], 'elu')
        init_op = tf.global_variables_initializer()
        merged_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(logdir='./test', graph=sess.graph)
        sess.run(init_op)

        # train
        for i in range(num_train):
            minibatch = samples[random.sample(range(samples.shape[0]), batch_size), :]
            sess.run(nnet.train_op, feed_dict={nnet.input_holder: minibatch, nnet.target: minibatch})

            if i % 100 == 0:
                summary = sess.run(merged_op, feed_dict={nnet.input_holder: minibatch, nnet.target: minibatch})
                summary_writer.add_summary(summary, global_step=i)

            if i % 1000 == 0:
                print('train %d' % (i + 1000), sess.run(nnet.loss(), feed_dict={nnet.input_holder: minibatch,
                                                                                nnet.target: minibatch}))

        P = sess.run(nnet.logits, feed_dict={nnet.input_holder: samples})
        xp = P[:, 0].reshape([x_grid.shape[0], x_grid.shape[1]])
        yp = P[:, 1].reshape([x_grid.shape[0], x_grid.shape[1]])
        zp = P[:, 2].reshape([x_grid.shape[0], x_grid.shape[1]])

        # visualization
        fig1 = plt.figure()
        ax = fig1.gca(projection='3d')
        ax.plot_surface(x_grid, y_grid, z0_grid,
                        rstride=1,  # row step size
                        cstride=1,  # column step size
                        cmap=cm.coolwarm,  # color map
                        linewidth=0,  # wireframe line width
                        antialiased=False)
        ax.set_title('Original')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.0, 2.5])
        # ax.colorbar

        fig2 = plt.figure(2)
        ax = fig2.gca(projection='3d')
        ax.plot_surface(x_grid, y_grid, z_grid, rstride=1, cstride=1,
                        cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_title('Input')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.0, 2.5])

        fig3 = plt.figure(3)
        ax = fig3.gca(projection='3d')
        ax.plot_surface(xp, yp, zp, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_title('Output-enc')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.0, 2.5])

        plt.show()
