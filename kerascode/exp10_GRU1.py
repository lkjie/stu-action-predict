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
from keras.layers import GRU, Input, Lambda
from keras.callbacks import TensorBoard, CSVLogger, EarlyStopping
import keras
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from keras.backend.tensorflow_backend import set_session

from kerascode.NNUtils import *
from kerascode.configure import *
from kerascode.NNoperator import run_model

'''
预测金额，通过学号的序列预测
'''

experiment = os.path.basename(__file__).replace('.py', '')
experiment = get_experiment_name(experiment)

print('Loading data...')

features = ['timeslot_week', ]
timeseries = ['student_id_int', 'timeslot_week', 'placei', 'amount']

feature_count = len(features)
timeseries_count = len(timeseries)
labels = ['amount']
label_cates = [consum[f].drop_duplicates().count() for f in labels]
emb_feat_cates = [consum[f].drop_duplicates().count() for f in features]
emb_feat_names = ['emb_feat_%s' % f for f in features]
emb_timeseries_cates = [consum[f].drop_duplicates().count() for f in timeseries]
emb_timeseries_names = ['emb_timeseries_%s' % f for f in timeseries]

xlist, currlist, ylist = load_data_expftl(features, timeseries, labels, 10)
if stratify:
    unique, counts = np.unique(ylist, return_counts=True)
    idy = np.isin(ylist, unique[counts > 1]).reshape(-1)
    ylist = ylist[idy]
    xlist = xlist[idy]
    currlist = currlist[idy]
    x_train1, x_test1, x_train2, x_test2, y_train, y_test = train_test_split(xlist, currlist, ylist, test_size=0.2,
                                                                             random_state=42, stratify=ylist)
else:
    x_train1, x_test1, x_train2, x_test2, y_train, y_test = train_test_split(xlist, currlist, ylist, test_size=0.2,
                                                                             random_state=42)

print(len(x_train1), 'train sequences')
print(len(x_test1), 'test sequences')


def build_model():
    print('Build model...')
    timeseries_inp = Input(shape=(timestep_len, timeseries_count), dtype='float32')
    branch_outputs = []
    for i in range(timeseries_count):
        out = Lambda(lambda x: x[:, :, i])(timeseries_inp)
        if timeseries[i] == 'student_id_int':
            nextlayer = Embedding(input_dim=emb_timeseries_cates[i], output_dim=12, input_length=timestep_len,
                                  mask_zero=False,
                                  trainable=True,
                                  name=emb_timeseries_names[i])(out)
        elif timeseries[i] == 'amount':
            nextlayer = Reshape(target_shape=(timestep_len, 1))(out)
        else:
            nextlayer = OneHot(input_dim=emb_timeseries_cates[i], input_length=timestep_len)(out)
        branch_outputs.append(nextlayer)
    timeseries_x = keras.layers.concatenate(branch_outputs)

    lstm1 = GRU(256, dropout=0.2, recurrent_dropout=0.2)(timeseries_x)

    branch_outputs = []
    fea_inp = Input(shape=(feature_count,), dtype='int32')
    for i in range(feature_count):
        out = Lambda(lambda x: x[:, i])(fea_inp)
        if features[i] == 'student_id_int':
            nextlayer = Embedding(input_dim=emb_feat_cates[i], output_dim=12, mask_zero=False,
                                  trainable=True,
                                  name=emb_feat_names[i])(out)
        elif features[i] == 'amount':
            nextlayer = Reshape(target_shape=(timestep_len, 1))(out)
        else:
            nextlayer = OneHot(input_dim=emb_feat_cates[i], input_length=1)(out)

        branch_outputs.append(nextlayer)

    branch_outputs.append(lstm1)
    merge1 = keras.layers.concatenate(branch_outputs)
    out = Dense(1)(merge1)

    model = Model(inputs=[timeseries_inp, fea_inp], outputs=[out])

    # try using different optimizers and different optimizer configs
    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['mse', 'mae'])
    return model


model = build_model()
run_model(experiment, model, [x_train1, x_train2], [y_train], [x_test1, x_test2], [y_test])
