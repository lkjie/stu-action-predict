#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'lkjie'

import pandas as pd
import pymysql, logging, os
import sys, datetime
from sqlalchemy import create_engine
import numpy as np
from dateutil.relativedelta import relativedelta
from scipy.stats import norm
import matplotlib.pyplot as plt
import queue, json, time, dateutil, math
from collections import namedtuple
import json
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding
from keras.layers import LSTM, Input, Lambda, Reshape
from keras.callbacks import TensorBoard, CSVLogger
import keras
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from keras.backend.tensorflow_backend import set_session

from kerascode.NNUtils import top1, top3, top5, top10, OneHot


'''
预测金额
'''

batch_size = 32
experiment = os.path.basename(__file__).replace('.py', '')

print('Loading data...')
consum = pd.read_csv('../data/consum_access_feat54.csv', nrows=200000)
consum['brush_time'] = pd.to_datetime(consum['brush_time'])
consum['timeslot_week'] = consum['dayofweek']*48 + consum['timeslot']
# features = ['amount', 'card_id', 'student_id_int', 'remained_amount', 'timeslot', 'placei', 'timeslot_week']
features = ['student_id_int', 'timeslot_week',
                # 'placei',
                # 'remained_amount',
                # 'trans_type',
                # 'category'
            ]
feature_count = len(features)
label = 'amount'

consum = consum[features + [label]]
label_cates = consum[label].drop_duplicates().count()
emb_cates = [consum[f].drop_duplicates().count() for f in features]
emb_names = ['embedding_%s'%f for f in features]

# x = consum[features].values
# y = consum[label].values

# x_train, x_test, y_train, y_test = train_test_split(consum[features], consum[label], test_size=0.2, random_state=42, stratify=consum[label])
x_train, x_test, y_train, y_test = train_test_split(consum[features], consum[label], test_size=0.2, random_state=42)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')


def build_model():
    print('Build model...')
    fea_inp = Input(shape=(feature_count, ), dtype='int32')
    branch_outputs = []
    for i in range(feature_count):
        # Slicing the ith channel:
        out = Lambda(lambda x: x[:, i])(fea_inp)
        out = Reshape((1,))(out)

        # Setting up your per-channel layers (replace with actual sub-models):
        emb = Embedding(input_dim=emb_cates[i], output_dim=100, input_length=1, mask_zero=False, trainable=True,
                        name=emb_names[i])(out)

        branch_outputs.append(emb)

    x = keras.layers.concatenate(branch_outputs)
    lstm1 = LSTM(1000, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(x)
    lstm2 = LSTM(200, dropout=0.2, recurrent_dropout=0.2)(lstm1)
    dense1 = Dense(50, activation='tanh')(lstm2)
    dense2 = Dense(1, activation='tanh')(dense1)

    model = Model(inputs=[fea_inp], outputs=[dense2])

    # try using different optimizers and different optimizer configs
    model.compile(loss='mean_squared_error',
                  optimizer='sgd',
                  metrics=['mae'])
    return model

model = build_model()
# keras_backend.set_session(tf_debug.TensorBoardDebugWrapperSession(tf.Session(), "localhost:6007"))
tensorboard = TensorBoard(log_dir='./%s_logs'%experiment,batch_size=batch_size,
                          embeddings_freq=5,
                          embeddings_layer_names=emb_names,
                          # embeddings_metadata='metadata.tsv',
                          embeddings_data=x_test)
csv_logger = CSVLogger('logs/%s_training.log'%experiment)
# gpu_options = tf.GPUOptions(allow_growth=True)
# config = tf.ConfigProto(gpu_options=gpu_options)
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config=config))

model.summary()
print('Train...')

model.fit(x_train, y_train,
          batch_size=batch_size,
          callbacks=[tensorboard, csv_logger],
          epochs=15,
          validation_data=(x_test, y_test))
eval_res = model.evaluate(x_test, y_test, batch_size=batch_size)
y_p = model.predict(x_test)
print('Test evaluation:')
print(model.metrics_names)
print(eval_res)
print(y_p)
model.save('models/%s_model'%experiment)
