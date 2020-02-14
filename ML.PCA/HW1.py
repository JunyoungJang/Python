import functions_from_Hvass as H
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_digits


class tensor_PCA:
    def __init__(self, data, target=None, dtype=tf.float32):
        self.data = data
        self.target = target
        self.dtype = dtype

        self.graph = None
        self.X = None
        self.u = None
        self.singular_values = None
        self.sigma = None
        self.trans_sigma = None
        self.pca = None
        self.mu = None

    def fit(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.X = tf.placeholder(self.dtype, shape=self.data.shape)

            # Perform SVD
            singular_values, u, _ = tf.svd(self.X)

            mu = tf.reduce_mean(self.X)

            sigma = tf.diag(singular_values)

        with tf.Session(graph=self.graph) as session:
            self.u, self.singular_values, self.mu, self.sigma = session.run([u, singular_values, mu, sigma],
                                                                   feed_dict={self.X: self.data})

    def reduce(self, n_dimensions=None, keep_info=None):
        if keep_info:
            # Normalize singular values
            normalized_singular_values = self.singular_values / sum(self.singular_values)

            # Create the aggregated ladder of kept information per dimension
            ladder = np.cumsum(normalized_singular_values)

            # Get the first index which is above the given information threshold
            index = next(idx for idx, value in enumerate(ladder) if value >= keep_info) + 1
            n_dimensions = index

        with self.graph.as_default():
            # Cut out the relevant part from sigma
            trans_sigma = tf.slice(self.sigma, [0, 0], [self.data.shape[1], n_dimensions])

            # PCA
            pca = tf.matmul(self.u, trans_sigma)

        with tf.Session(graph=self.graph) as session:
            self.trans_sigma, self.pca = session.run([trans_sigma, pca], feed_dict={self.X: self.data})
            return pca

    def recovery(self):
        with self.graph.as_default():
            inv_pca = tf.add(tf.matmul(self.pca, tf.transpose(self.trans_sigma)), self.mu)
        with tf.Session(graph=self.graph) as session:
            return session.run(inv_pca, feed_dict={self.X: self.data})


digits = load_digits()
data = digits.data
data = data/255.
data = data - data.mean(axis=0)
y = digits.target

# original image
img = data[0,:]
img_taget = y[0]
img_shape = (8, 8)

tf_pca = tensor_PCA(data, y)
tf_pca.fit()
tran_pca = tf_pca.reduce(n_dimensions=3)
inv_pca = tf_pca.recovery()

im = inv_pca[0, :]
H.plot_image(im, img_shape)
