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
from keras.layers import LSTM, Input, GRU
from keras.callbacks import TensorBoard, CSVLogger, EarlyStopping
import keras
from keras.layers import Lambda
# We will use `one_hot` as implemented by one of the backends

import tensorflow as tf
from tensorflow.python import debug as tf_debug
from keras.backend.tensorflow_backend import set_session

from kerascode.NNUtils import *
from kerascode.configure import *
from kerascode.NNoperator import run_model

'''
预测金额
'''

experiment = os.path.basename(__file__).replace('.py', '')
experiment = get_experiment_name(experiment)

print('Loading data...')

features = ['amount', 'card_id', 'student_id_int', 'remained_amount', 'timeslot']
features = ['student_id_int', 'timeslot_week',
            # 'amount',
            # 'remained_amount',
            # 'trans_type',
            # 'category'
            ]
labels = ['placei']

timeslot_cates = consum['timeslot_week'].drop_duplicates().count()
student_id_cates = consum['student_id_int'].drop_duplicates().count()
label_cates = [consum[f].drop_duplicates().count() for f in labels]

student_id_emb_data = consum[features].drop_duplicates(subset=['student_id_int'])
student_id_emb_data = [student_id_emb_data['student_id_int'], student_id_emb_data['timeslot_week']]

if stratify:
    x_train, x_test, y_train, y_test = train_test_split(consum[features], consum[labels], test_size=0.2,
                                                        random_state=42,
                                                        stratify=consum[labels])
else:
    x_train, x_test, y_train, y_test = train_test_split(consum[features], consum[labels], test_size=0.2,
                                                        random_state=42)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

x = [x_train['student_id_int'], x_train['timeslot_week']]
x_t = [x_test['student_id_int'], x_test['timeslot_week']]


def build_model():
    print('Build model...')
    id_inp = Input(shape=(1,), dtype='int32')
    id_emb = Embedding(input_dim=student_id_cates, output_dim=12, input_length=1, mask_zero=False, trainable=True,
                       name='student_id')(id_inp)
    time_inp = Input(shape=(1,), dtype='int32')
    time_onehot = OneHot(input_dim=timeslot_cates, input_length=1)(time_inp)

    x = keras.layers.concatenate([id_emb, time_onehot])
    lstm1 = GRU(512, dropout=0.2, recurrent_dropout=0.2)(x)
    out = Dense(label_cates[0], activation='softmax')(lstm1)

    model = Model(inputs=[id_inp, time_inp], outputs=[out])

    # try using different optimizers and different optimizer configs
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=[top1, top3, top5, top10])
    return model


model = build_model()
run_model(experiment, model, x, y_train, x_t, y_test, batch_size=batch_size, epochs=epochs)
