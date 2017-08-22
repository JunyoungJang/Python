from __future__ import division, print_function, absolute_import
import pandas as pd
import numpy as np
import tflearn
from sklearn.cross_validation import train_test_split
from tflearn.data_utils import to_categorical
from tflearn.metrics import Accuracy

# Import data
RawData = pd.read_excel('data.xlsx', sheetname='train')
Data = np.array(RawData.iloc[0:1858, 1:4])
allX = np.array(Data[0:len(Data), 0:2])
allY = np.array(Data[0:len(Data), 2])
trainX, testX, trainY, testY = train_test_split(allX, allY, test_size=0.1, random_state=42)
trainY = to_categorical(np.array(trainY), nb_classes=3)
testY = to_categorical(np.array(testY), nb_classes=3)

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
net = tflearn.regression(softmax, optimizer='adam', learning_rate=0.0005, metric=acc, loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0, checkpoint_path='RiskCriterion.tflearn', max_checkpoints=1)
model.fit(trainX, trainY, n_epoch=5000, validation_set=(testX, testY), shuffle=True, show_metric=True,
          run_id='RiskCriterion')
model.save('RiskCriterion.tflearn')