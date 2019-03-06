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
同时预测地点、金额、时间
'''

experiment = os.path.basename(__file__).replace('.py', '')
experiment = get_experiment_name(experiment)

print('Loading data...')

timeseries = ['student_id_int', 'timeslot_week', 'placei', 'amount']

timeseries_count = len(timeseries)
labels = ['timeslot_week']
label_cates = [consum[f].drop_duplicates().count() for f in labels]
emb_timeseries_cates = [consum[f].drop_duplicates().count() for f in timeseries]
emb_timeseries_names = ['emb_timeseries_%s' % f for f in timeseries]

xlist, ylist = load_data_exptl(timeseries, labels, 13)
if stratify:
    unique, counts = np.unique(ylist, return_counts=True)
    idy = np.isin(ylist, unique[counts > 1]).reshape(-1)
    ylist = ylist[idy]
    xlist = xlist[idy]
    x_train1, x_test1, y_train1, y_test1 = train_test_split(xlist, ylist, test_size=0.2,
                                                            random_state=42,
                                                            stratify=ylist)
else:
    x_train1, x_test1, y_train1, y_test1 = train_test_split(xlist, ylist,
                                                            test_size=0.2,
                                                            random_state=42)

print(len(x_train1), 'train sequences')
print(len(x_test1), 'test sequences')


def sparse_focal_loss(y_true, y_pred):
    '''
    多标签分类的focal_loss，输入target_tensor为一个正整数，表示类别
    :param prediction_tensor:
    :param target_tensor:
    :param weights:
    :param alpha:
    :param gamma:
    :return:
    '''
    y_true = tf.reshape(y_true, [-1])
    y_true = tf.cast(y_true, dtype='int64')
    y_true = tf.one_hot(y_true, label_cates[0])
    res = focal_loss_noalpha(y_pred, y_true)
    return res


def time_loss(y_true, y_pred):
    '''
    超过24小时的loss计算为24小时
    :param prediction_tensor:
    :param target_tensor:
    :param weights:
    :param alpha:
    :param gamma:
    :return:
    '''
    y_pred = array_ops.where(y_pred - y_true > 24, y_true + 24, y_pred)
    y_pred = array_ops.where(y_pred - y_true < -24, y_true + 24, y_pred)
    res = K.mean(K.square(y_pred - y_true), axis=-1)
    return res


def build_model():
    print('Build model...')
    timeseries_inp = Input(shape=(timestep_len, timeseries_count), dtype='float32')
    branch_outputs = []
    for i in range(timeseries_count):
        out = Lambda(lambda x: x[:, :, i])(timeseries_inp)
        if timeseries[i] == 'student_id_int':
            # out = K.cast(out, dtype='int32')
            nextlayer = Embedding(input_dim=emb_timeseries_cates[i], output_dim=12, input_length=timestep_len,
                                  mask_zero=False,
                                  trainable=True,
                                  name=emb_timeseries_names[i])(out)
        elif timeseries[i] == 'amount':
            nextlayer = Reshape(target_shape=(timestep_len, 1))(out)
        else:
            # out = K.cast(out, dtype='int32')
            nextlayer = OneHot(input_dim=emb_timeseries_cates[i], input_length=timestep_len)(out)
        branch_outputs.append(nextlayer)
    timeseries_x = keras.layers.concatenate(branch_outputs)

    lstm1 = GRU(256, dropout=0.2, recurrent_dropout=0.2)(timeseries_x)

    # out_place = Dense(label_cates[0], activation='softmax', name='out_place')(lstm1)
    out_time = Dense(label_cates[0], activation='tanh')(lstm1)
    out_time = Dense(1, activation='relu', name='out_time')(out_time)

    model = Model(inputs=[timeseries_inp], outputs=[out_time])

    # try using different optimizers and different optimizer configs
    model.compile(loss=[time_loss],
                  optimizer='adam',
                  metrics={'out_time': ['mse', 'mae']})
    return model


model = build_model()

run_model(experiment, model, [x_train1], [y_train1], [x_test1], [y_test1])
