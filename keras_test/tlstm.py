'''Trains an LSTM model on the IMDB sentiment classification task.

The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.

# Notes

- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.

- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
from __future__ import print_function
import sys
import os
import numpy as np
# print(os.environ)
# print(sys.path)
# os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-9.0/lib64:' + os.environ['LD_LIBRARY_PATH']
# sys.path.append('/usr/local/cuda-9.0/lib64')
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM
from keras.datasets import imdb
from keras.callbacks import TensorBoard

max_features = 20000
# cut texts after this number of words (among top max_features most common words)
batch_size = 32
epochs = 5

x_train = np.random.rand(1000, 1, 1)
y_train = np.random.rand(1000, 1)

print('Build model...')
model = Sequential()
l = LSTM(128, dropout=0.2, return_sequences=False, input_shape=(1, 1), implementation=2)
model.add(l)
model.add(Dense(1))
# try using different optimizers and different optimizer configs
model.compile(loss='mse',
              optimizer='adam',
              metrics=['mse'])
model.summary()
print('Train...')

tensorboard = TensorBoard(log_dir='./test_logs', batch_size=batch_size)
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=[tensorboard])