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
from keras.callbacks import TensorBoard, CSVLogger, EarlyStopping
import keras
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from keras.backend.tensorflow_backend import set_session

sys.path.append('/home/liwenjie/liwenjie/projects/lwjpaper')
from kerascode.NNUtils import *
from kerascode.configure import *
from kerascode.NNoperator import run_model

'''
预测金额
'''

experiment = os.path.basename(__file__).replace('.py', '')
experiment = get_experiment_name(experiment)

print('Loading data...')

features = ['student_id_int', 'timeslot_week',
            'placei',
            # 'remained_amount',
            # 'trans_type',
            # 'category'
            ]
feature_count = len(features)
labels = ['amount']

label_cates = [consum[f].drop_duplicates().count() for f in labels]
emb_cates = [consum[f].drop_duplicates().count() for f in features]
emb_names = ['embedding_%s' % f for f in features]

# 金额不需要stratify
x_train, x_test, y_train, y_test = train_test_split(consum[features], consum[labels], test_size=0.2, random_state=42)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')


def build_model():
    print('Build model...')
    fea_inp = Input(shape=(feature_count,), dtype='int32')
    branch_outputs = []
    for i in range(feature_count):
        # Slicing the ith channel:
        out = Lambda(lambda x: x[:, i])(fea_inp)
        out = Reshape((1,))(out)

        if features[i] == 'student_id_int':
            nextlayer = Embedding(input_dim=emb_cates[i], output_dim=12, input_length=1, mask_zero=False,
                                  trainable=True,
                                  name=emb_names[i])(out)
        else:
            nextlayer = OneHot(input_dim=emb_cates[i], input_length=1)(out)

        branch_outputs.append(nextlayer)

    x = keras.layers.concatenate(branch_outputs)
    lstm1 = LSTM(1024, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(x)
    lstm2 = LSTM(256, dropout=0.2, recurrent_dropout=0.2)(lstm1)
    dense1 = Dense(50, activation='tanh')(lstm2)
    dense2 = Dense(1, activation='tanh')(dense1)

    model = Model(inputs=[fea_inp], outputs=[dense2])

    # try using different optimizers and different optimizer configs
    model.compile(loss='mean_squared_error',
                  optimizer='sgd',
                  metrics=['mae'])
    return model


model = build_model()

run_model(experiment, model, x_train, y_train, x_test, y_test, batch_size=batch_size, epochs=epochs)
