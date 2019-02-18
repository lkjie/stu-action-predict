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
from keras.layers import LSTM, Input
from keras.callbacks import TensorBoard, CSVLogger
import keras
from keras.layers import Lambda
# We will use `one_hot` as implemented by one of the backends
from keras import backend as K
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
consum = pd.read_csv('../data/consum_access_feat54.csv', nrows=500000)
consum['brush_time'] = pd.to_datetime(consum['brush_time'])
consum['timeslot_week'] = consum['dayofweek']*48 + consum['timeslot']
features = ['amount', 'card_id', 'student_id_int', 'remained_amount', 'timeslot']
features = ['student_id_int', 'timeslot_week',
                # 'amount',
                # 'remained_amount',
                # 'trans_type',
                # 'category'
            ]
label = 'placei'

consum = consum[features + [label]]
timeslot_cates = consum['timeslot_week'].drop_duplicates().count()
student_id_cates = consum['student_id_int'].drop_duplicates().count()
label_cates = consum[label].drop_duplicates().count()

# x_train, x_test, y_train, y_test = train_test_split(consum[features], consum[label], test_size=0.2, random_state=42, stratify=consum[label])
x_train, x_test, y_train, y_test = train_test_split(consum[features], consum[label], test_size=0.2, random_state=42)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

x = [x_train['student_id_int'], x_train['timeslot_week']]
x_t = [x_test['student_id_int'], x_test['timeslot_week']]


def build_model():
    print('Build model...')
    id_inp = Input(shape=(1,), dtype='int32')
    id_emb = Embedding(input_dim=student_id_cates, output_dim=100, input_length=1, mask_zero=False, trainable=True, name='student_id')(id_inp)
    time_inp = Input(shape=(1,), dtype='int32')
    time_emb = Embedding(input_dim=timeslot_cates, output_dim=100, input_length=1, mask_zero=False, trainable=True, name='timeslot_week')(time_inp)

    x = keras.layers.concatenate([id_emb, time_emb])
    lstm1 = LSTM(1000, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(x)
    lstm2 = LSTM(200, dropout=0.2, recurrent_dropout=0.2)(lstm1)
    out = Dense(label_cates, activation='softmax')(lstm2)

    model = Model(inputs=[id_inp, time_inp], outputs=[out])

    # try using different optimizers and different optimizer configs
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', top1, top3, top5, top10])
    return model

model = build_model()
# keras_backend.set_session(tf_debug.TensorBoardDebugWrapperSession(tf.Session(), "localhost:6007"))
tensorboard = TensorBoard(log_dir='./%s_logs'%experiment,batch_size=batch_size,
                          embeddings_freq=5,
                          embeddings_layer_names=['student_id', 'timeslot_week'],
                          # embeddings_metadata='metadata.tsv',
                          embeddings_data=x_t)
csv_logger = CSVLogger('logs/%s_training.log'%experiment)
# gpu_options = tf.GPUOptions(allow_growth=True)
# config = tf.ConfigProto(gpu_options=gpu_options)
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config=config))
model.summary()
print('Train...')

model.fit(x, y_train,
          batch_size=batch_size,
          callbacks=[tensorboard, csv_logger],
          epochs=10,
          validation_data=(x_t, y_test))
eval_res = model.evaluate(x_t, y_test, batch_size=batch_size)
y_p = model.predict(x_t)
print('Test evaluation:')
print(model.metrics_names)
print(eval_res)
print(y_p)
model.save('models/%s_model'%experiment)
