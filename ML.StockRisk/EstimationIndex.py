from __future__ import division, print_function, absolute_import
import pandas as pd
import numpy as np
import tflearn
from sklearn.cross_validation import train_test_split
from tflearn.data_utils import to_categorical
from tflearn.metrics import Accuracy

# Import data
RawData = pd.read_excel('data2.xlsx', sheetname='train')
Data = np.array(RawData.iloc[0:1112, 1:15])
allX = np.array(Data[0:len(Data), 0:13])
allY = np.array(Data[0:len(Data), 13])

trainX, testX, trainY, testY = train_test_split(allX, allY, test_size=0.1, random_state=42)
trainY = to_categorical(np.array(trainY), nb_classes=2)
testY = to_categorical(np.array(testY), nb_classes=2)

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
net = tflearn.regression(softmax, optimizer='adam', learning_rate=0.1, metric=acc,
                             loss='categorical_crossentropy')
# Training
model = tflearn.DNN(net, tensorboard_verbose=0, checkpoint_path='EstimationIndex.tflearn', max_checkpoints=1)
model.fit(trainX, trainY, n_epoch=15, validation_set=(testX, testY), shuffle=True, show_metric=True,
          run_id='EstimationIndex')
model.save('EstimationIndex.tflearn')