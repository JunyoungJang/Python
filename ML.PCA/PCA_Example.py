import tensorflow as tf
import numpy as np



def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


def intercept_variable():
    b = tf.Variable(tf.zeros([1]))

    return b


class Layer():

    def __repr__(self):
        act = str(self.transfer_function).split(' ')[1]
        rep = 'Layer. n_nodes:' + str(self.n_input) + ' act:' + act + \
              ' n_output:' + str(self.n_output)
        return rep

    def __init__(self, input, n_input, n_output,
                 transfer_function=None, weights=None,
                 b=None):

        self.input = input
        self.n_input = n_input
        self.n_output = n_output

        if transfer_function is None:
            self.transfer_function = tf.nn.relu
        else:
            self.transfer_function = transfer_function

        if weights is None:
            self.weights = tf.Variable(xavier_init(self.n_input, self.n_output))
        else:
            self.weights = weights

        if b is None:
            self.b = tf.Variable(tf.zeros([self.n_output], dtype=tf.float32))
        else:
            self.b = b

        self.lin_output = tf.add(tf.matmul(self.input, self.weights), self.b)
        self.output = self.transfer_function(self.lin_output)

        return



import sklearn as sk
from tensorflow.examples.tutorials.mnist import input_data
import sklearn.preprocessing


class Autoencoder():

    def __init__(self, nodes, activactions):

        self.activactions = activactions
        encoding_nodes = nodes[:-1]
        latent_dimension = [nodes[-1]]
        decoding_nodes = encoding_nodes.copy()
        decoding_nodes.reverse()
        self.nodes = encoding_nodes + latent_dimension + decoding_nodes
        self.n_layers = len(nodes)
        print('Nodes = ', self.nodes)

        self._get_encoders(encoding_nodes)
        self._get_decoders(latent_dimension, decoding_nodes)
        self.layers = self.encoding_layers +  self.decoding_layers

        return

    def _get_encoders(self, encoding_nodes):

        self.x = tf.placeholder(tf.float32, shape=[None, self.nodes[0]], name='X')
        l = 0
        self.encoding_layers = [Layer(input=self.x, transfer_function=self.activactions[l],
                                      weights=None,
                                      b=None,
                                      n_input=self.nodes[0],
                                      n_output=self.nodes[1])] \
                                      + [None]*(len(encoding_nodes) - 1)
        l += 1
        for i, node in enumerate(encoding_nodes[1:]):
            new_layer = Layer(input=self.encoding_layers[i].output,
                              transfer_function=self.activactions[i+1],
                              n_input=self.nodes[i+1],
                              n_output=self.nodes[i+2])
            self.encoding_layers[i + 1] = new_layer

        return


    def _get_decoders(self, latent_dimension, decoding_nodes):
        l = len(self.encoding_layers)
        t_weights = tf.transpose(self.encoding_layers[-1].weights)
        self.decoding_layers = [Layer(input=self.encoding_layers[-1].output,
                                      transfer_function=self.activactions[l],
                                      n_input=latent_dimension[0],
                                      n_output=decoding_nodes[0],
                                      weights=t_weights)]
        l += 1
        for i, node in enumerate(decoding_nodes[1:]):
            t_weights = tf.transpose(self.encoding_layers[-i-2].weights)
            new_layer = Layer(input=self.decoding_layers[i].output,
                              transfer_function=self.activactions[l],
                              n_input=decoding_nodes[i],
                              n_output=decoding_nodes[i+1],
                              weights=t_weights)
            self.decoding_layers.append(new_layer)
            l += 1

        return

    def _build_network(self):
        return


def mse(x, x_est):

    return np.linalg.norm(x - x_est)/np.linalg.norm(x)


def data1():
    n_samples, n_features = 2000, 5
    X = np.random.uniform(0, 1, (n_samples, n_features))
    X[:, 2] = X[:, 1]**3
    X[:, 3] = X[:, 1]*X[:, 2]
    X[:, 4] = X[:,1]**2 * X[:, 0]**3
    X = sk.preprocessing.scale(X)
    return X

def mnist():
    DG = DataGenerator()
    file_path = DG.load_path + 'MNIST'

    mnist = input_data.read_data_sets(file_path, one_hot=True)
    X = mnist.train.images
    unique = np.apply_along_axis(lambda x: len(np.unique(x)), 0, X)

    X = X[:, unique != 1]
    n_samples, n_features = X.shape

    mu = X.mean(axis=0)
    X = X - mu
    return X



def pca_err(X):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2).fit(X)
    x_pca = pca.transform(X)
    lr = sk.linear_model.LinearRegression().fit(x_pca, X)
    x_est = lr.predict(x_pca)
    print('err pca = ', mse(X, x_est))
    return

np.random.seed(1)

X = data1()
#X = mnist()
n_samples, n_features = X.shape

pca_err(X)

nodes = [n_features, 20, 10, 2]
activactions = [tf.identity] + [tf.nn.relu] * 6

aut = Autoencoder(nodes=nodes, activactions=activactions)
layers = aut.layers

input_nn = layers[0].input
output_nn = layers[-1].output
cost = tf.nn.l2_loss(tf.sub(input_nn, output_nn))


start_learning_rate = 0.02
learning_rate = tf.Variable(start_learning_rate, trainable=False)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer(start_learning_rate).minimize(cost)


saver = tf.train.Saver()


sess = tf.InteractiveSession()
init = tf.initialize_all_variables()
sess.run(init)



iterations = 100000000
batch_size = 200
j = 0
for i in range(iterations):
    rand_learning_rate = np.random.uniform(0, 1)
    batch_index = list(range(j, j+batch_size))
    j += batch_size
    if j+batch_size >= n_samples:
        j = 0
    optimizer.run(feed_dict={input_nn: X[batch_index]})

    if i % 10000 == 0:
        x_est = output_nn.eval(session=sess, feed_dict={input_nn: X})
        print('it=', i, 'learning_rate=', learning_rate.eval(session=sess),
               ' mse=', mse(X, x_est))

    if i % 100000 == 0 :
        learning_rate /= 2


save_path = saver.save(sess, "autoencoder.ckpt")