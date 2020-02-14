import tensorflow as tf, numpy as np, matplotlib.pyplot as plt
import random
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


class NeuralNet:
    def __init__(self, n_in, n_out, n_hid, act_ft, lr, bn_lr):
        """
        :param n_in: input dimension
        :param n_out: output dimension
        :param n_hid: list, hidden layer
        :param act_ft: string, activated function
        :param lr: float, learning rate
        :param bn_lr: float, learning rate of batch normalization
        """
        self.n_in = n_in
        self.n_out = n_out
        self.n_hid = n_hid  # list
        self.act_ft = act_ft
        self.lr = lr
        self.bn_lr = bn_lr

        self.num_layer = len(self.n_hid)

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
        self.phase_train = tf.placeholder(dtype=tf.bool, name='phase')
        self.input_holder = tf.placeholder(dtype=tf.float32, name='input')
        self.target = tf.placeholder(dtype=tf.float32, name='target')
        self.output = [self.input_holder]

        # train
        self.logits = self.inference()
        self.opt = tf.train.AdamOptimizer(self.lr)  # default 0.001
        self.train_op = self.opt.minimize(self.loss())

    def weight_var(self, n_in, n_out):
        weight = tf.get_variable(name='weight', shape=[n_in, n_out],
                                 initializer=tf.contrib.layers.xavier_initializer())
        tf.summary.histogram('weight', weight)
        return weight

    def bias_var(self, n_out, b_init):
        bias = tf.get_variable(name='bias', initializer=tf.constant(b_init, shape=[n_out]))
        tf.summary.histogram('bias', bias)
        return bias

    def inference(self):

        # hidden layer
        with tf.variable_scope('layer0'):
            W = self.weight_var(self.n_in, self.n_hid[0])
            b = self.bias_var(self.n_hid[0], self.b_init)
            output = self.act(tf.matmul(self.output[-1], W) + b)
            # pre = tf.matmul(self.output[-1], W)
            # normed1, normed2 = self.batch_norm(pre, self.n_hid[0], self.bn_lr)
            # normed = tf.where(self.phase_train, normed1, normed2)
            # output = self.act(normed)
            self.output.append(output)

        for i in range(self.num_layer-1):
            with tf.variable_scope('layer'+str(i+1)):
                W = self.weight_var(self.n_hid[i], self.n_hid[i+1])
                b = self.bias_var(self.n_hid[i+1], self.b_init)
                output = self.act(tf.matmul(self.output[-1], W) + b)
                # pre = tf.matmul(self.output[-1], W)
                # normed1, normed2 = self.batch_norm(pre, self.n_hid[i+1], self.bn_lr)
                # normed = tf.where(self.phase_train, normed1, normed2)
                # output = self.act(normed)
                self.output.append(output)

        # output layer
        with tf.variable_scope('output'):
            W = self.weight_var(self.n_hid[-1], self.n_out)
            b = self.bias_var(self.n_out, 0.00)
            output = self.act(tf.matmul(self.output[-1], W) + b)
            self.output.append(output)
            return self.output[-1]

    def loss(self):
        loss = tf.reduce_mean(tf.square(self.logits - self.target), name='loss')
        tf.summary.scalar('loss', loss)
        return loss

    def batch_norm(self, t, n_out, bn_lr):

        with tf.variable_scope('bn'):
            scale = tf.Variable(tf.constant(1.0, shape=[n_out]), name='scale')
            offset = tf.Variable(tf.constant(0.0, shape=[n_out]), name='offset')
            ema_mean = tf.Variable(tf.constant(0.0, shape=[n_out]), name='ema_mean')
            ema_var = tf.Variable(tf.constant(1.0, shape=[n_out]), name='ema_var')

            batch_mean, batch_var = tf.nn.moments(t, axes=[0], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=1.0 - bn_lr)
            ema_apply_op = ema.apply([batch_mean, batch_var])
            ema_mean_var_assign_op = [ema_mean.assign(ema.average(batch_mean)),
                                      ema_var.assign(ema.average(batch_var))]

            with tf.control_dependencies([ema_apply_op]):
                with tf.control_dependencies(ema_mean_var_assign_op):
                    bm = tf.identity(batch_mean)
                    bv = tf.identity(batch_var)
                    em = tf.identity(ema_mean)
                    ev = tf.identity(ema_var)

            normed1 = tf.nn.batch_normalization(t, bm, bv, offset, scale, variance_epsilon=1e-3)
            normed2 = tf.nn.batch_normalization(t, em, ev, offset, scale, variance_epsilon=1e-3)

            tf.summary.histogram('offset', offset)
            tf.summary.histogram('scale', scale)
            tf.summary.histogram('moving_avg_mean', ema_mean)
            tf.summary.histogram('moving_avg_var', ema_var)

            return normed1, normed2



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
        nnet = NeuralNet(3, 3, [10, 10, 2, 10, 10], 'elu', 0.001, 0.001)
        init_op = tf.global_variables_initializer()
        merged_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(logdir='./test2', graph=sess.graph)
        sess.run(init_op)

        # train
        for i in range(num_train):
            minibatch = samples[random.sample(range(samples.shape[0]), batch_size), :]
            sess.run(nnet.train_op, feed_dict={nnet.input_holder: minibatch,
                                               nnet.target: minibatch, nnet.phase_train: True})

            if i % 100 == 0:
                summary = sess.run(merged_op, feed_dict={nnet.input_holder: minibatch,
                                                         nnet.target: minibatch, nnet.phase_train: True})
                summary_writer.add_summary(summary, global_step=i)

            if i % 1000 == 0:
                print('train %d' % (i + 1000), sess.run(nnet.loss(),
                                                        feed_dict={nnet.input_holder: minibatch,
                                                                   nnet.target: minibatch, nnet.phase_train: True}))

        P = sess.run(nnet.logits, feed_dict={nnet.input_holder: samples, nnet.phase_train: False})
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
        ax.set_title('Output')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.0, 2.5])

        plt.show()
