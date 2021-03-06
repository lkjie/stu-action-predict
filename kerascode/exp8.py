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
from keras.layers import Dense, Embedding, Reshape
from keras.layers import LSTM, Input, Lambda
from keras.callbacks import TensorBoard, CSVLogger, EarlyStopping
import keras
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from keras.backend.tensorflow_backend import set_session

from kerascode.NNUtils import *
from kerascode.configure import *
from kerascode.NNoperator import run_model

'''
预测地点，通过学号的序列预测序列
'''

experiment = os.path.basename(__file__).replace('.py', '')
experiment = get_experiment_name(experiment)

print('Loading data...')

# features = ['amount', 'card_id', 'student_id_int', 'remained_amount', 'timeslot']
features = ['student_id_int', 'timeslot_week',
            # 'amount',
            # 'remained_amount',
            # 'trans_type',
            # 'category'
            ]
timeseries = ['timeslot_week', 'placei', 'student_id_int']

feature_count = len(features)
timeseries_count = len(timeseries)
labels = ['placei']
label_cates = [consum[f].drop_duplicates().count() for f in labels]
# emb_feat_cates = [consum[f].drop_duplicates().count() for f in features]
# emb_feat_names = ['emb_feat_%s'%f for f in features]
emb_timeseries_cates = [consum[f].drop_duplicates().count() for f in timeseries]
emb_timeseries_names = ['emb_timeseries_%s' % f for f in timeseries]

xlist, ylist = load_data_exptl(timeseries, labels, 8)
if stratify:
    unique, counts = np.unique(ylist, return_counts=True)
    idy = np.isin(ylist, unique[counts > 1]).reshape(-1)
    ylist = ylist[idy]
    xlist = xlist[idy]
    x_train, x_test, y_train, y_test = train_test_split(xlist, ylist, test_size=0.2, random_state=42, stratify=ylist)
else:
    x_train, x_test, y_train, y_test = train_test_split(xlist, ylist, test_size=0.2, random_state=42)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')


def build_model():
    print('Build model...')
    timeseries_inp = Input(shape=(timestep_len, timeseries_count), dtype='int32')
    branch_outputs = []
    for i in range(timeseries_count):
        # Slicing the ith channel:
        out = Lambda(lambda x: x[:, :, i])(timeseries_inp)

        if timeseries[i] == 'student_id_int':
            nextlayer = Embedding(input_dim=emb_timeseries_cates[i], output_dim=12, input_length=timestep_len,
                                  mask_zero=False,
                                  trainable=True,
                                  name=emb_timeseries_names[i])(out)
        else:
            nextlayer = OneHot(input_dim=emb_timeseries_cates[i], input_length=timestep_len)(out)

        branch_outputs.append(nextlayer)
    timeseries_x = keras.layers.concatenate(branch_outputs)

    lstm1 = LSTM(1024, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(timeseries_x)
    lstm2 = LSTM(256, dropout=0.2, recurrent_dropout=0.2)(lstm1)
    out = Dense(label_cates[0], activation='softmax')(lstm2)

    model = Model(inputs=[timeseries_inp], outputs=[out])

    # try using different optimizers and different optimizer configs
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=[top1, top3, top5, top10])
    return model


model = build_model()
run_model(experiment, model, x_train, y_train, x_test, y_test, batch_size=batch_size, epochs=epochs)
