from __future__ import division, print_function, absolute_import
import tensorflow as tf
import tflearn
from tflearn.metrics import Accuracy

def RiskZone_Criterion(X):
    tf.reset_default_graph()
    tflearn.config.init_training_mode()
    input_layer = tflearn.input_data(shape=[None, 2])
    network = tflearn.fully_connected(input_layer, 256, activation='relu')
    network = tflearn.dropout(network, 0.5)
    network = tflearn.fully_connected(network, 512, activation='relu')
    network = tflearn.dropout(network, 0.5)
    network = tflearn.fully_connected(network, 256, activation='relu')
    network = tflearn.dropout(network, 0.5)
    network = tflearn.fully_connected(network, 512, activation='relu')
    network = tflearn.dropout(network, 0.5)
    network = tflearn.fully_connected(network, 1024, activation='relu')
    network = tflearn.dropout(network, 0.5)
    network = tflearn.fully_connected(network, 128, activation='relu')
    network = tflearn.dropout(network, 0.5)
    softmax = tflearn.fully_connected(network, 3, activation='softmax')

    acc = Accuracy(name="Accuracy")
    net = tflearn.regression(softmax, optimizer='adam', learning_rate=0.0005, metric=acc,
                             loss='categorical_crossentropy')

    model = tflearn.DNN(net)
    model.load('RiskCriterion.tflearn')
    model_result = model.predict(X)
    return model_result

def Estimation_Index_updown(X):
    tf.reset_default_graph()
    tflearn.config.init_training_mode()
    input_layer = tflearn.input_data(shape=[None, 13])
    network = tflearn.fully_connected(input_layer, 1024, activation='tanh')
    network = tflearn.fully_connected(network, 512, activation='tanh')
    network = tflearn.fully_connected(network, 512, activation='tanh')
    network = tflearn.fully_connected(network, 1024, activation='tanh')
    network = tflearn.fully_connected(network, 2048, activation='tanh')
    network = tflearn.fully_connected(network, 512, activation='tanh')
    network = tflearn.dropout(network, 0.7)
    softmax = tflearn.fully_connected(network, 2, activation='softmax')

    acc = Accuracy(name="Accuracy")
    net = tflearn.regression(softmax, optimizer='adam', learning_rate=0.01, metric=acc,
                             loss='categorical_crossentropy')

    model = tflearn.DNN(net)
    model.load('EstimationIndex.tflearn')
    model_result = model.predict(X)
    return model_result