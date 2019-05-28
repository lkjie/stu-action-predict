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

sys.path.append('/home/liwenjie/liwenjie/projects/lwjpaper')

from kerascode.NNUtils import *
from kerascode.configure import *
from kerascode.NNoperator import run_model

'''
预测地点，预测下两次的刷卡地点，通过学号的序列预测序列
'''
# features = ['amount', 'card_id', 'student_id_int', 'remained_amount', 'timeslot']
features = ['timeslot_week',
            # 'amount',
            # 'remained_amount',
            # 'trans_type',
            # 'category'
            ]
timeseries = ['student_id_int', 'timeslot_week', 'placei']

feature_count = len(features)
timeseries_count = len(timeseries)
labels = ['placei']
label_cates = [consum[f].drop_duplicates().count() for f in labels]
emb_feat_cates = [consum[f].drop_duplicates().count() for f in features]
emb_feat_names = ['emb_feat_%s' % f for f in features]
emb_timeseries_cates = [consum[f].drop_duplicates().count() for f in timeseries]
emb_timeseries_names = ['emb_timeseries_%s' % f for f in timeseries]

def build_model():
    print('Build model...')
    timeseries_inp = Input(shape=(timestep_len, timeseries_count), dtype='int32')
    branch_outputs = []
    for i in range(timeseries_count):
        out = Lambda(lambda x: x[:, :, i])(timeseries_inp)
        if timeseries[i] == 'student_id_int':
            nextlayer = Embedding(input_dim=emb_timeseries_cates[i], output_dim=32, input_length=timestep_len,
                                  mask_zero=False,
                                  trainable=True,
                                  name=emb_timeseries_names[i])(out)
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
            nextlayer = Embedding(input_dim=emb_feat_cates[i], output_dim=32, mask_zero=False,
                                  trainable=True,
                                  name=emb_feat_names[i])(out)
        else:
            nextlayer = OneHot(input_dim=emb_feat_cates[i], input_length=1)(out)

        branch_outputs.append(nextlayer)

    branch_outputs.append(lstm1)
    merge1 = keras.layers.concatenate(branch_outputs)
    out = Dense(label_cates[0], activation='softmax')(merge1)

    model = Model(inputs=[timeseries_inp, fea_inp], outputs=[out])

    # try using different optimizers and different optimizer configs
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=[top1, top3, top5, top10])
    return model


def main(label_location, timestep_split_len):
    experiment = os.path.basename(__file__).replace('.py', '')
    experiment = get_experiment_name(experiment) + '_labelLocation%d_timestepSplitLen%d' % (
    label_location, timestep_split_len)

    print('timestep_len: %d, label_location: %d, timestep_split_len: %d' % (
    timestep_len, label_location, timestep_split_len))
    print('Loading data...')
    x_train1, x_test1, x_train2, x_test2, y_train, y_test = load_data_expftl_train_test(features, timeseries, labels, 17, label_location=label_location, timestep_split_len=timestep_split_len)

    print(len(x_train1), 'train sequences')
    print(len(x_test1), 'test sequences')
    model = build_model()
    run_model(experiment, model, [x_train1, x_train2], [y_train], [x_test1, x_test2], [y_test], batch_size=batch_size, epochs=epochs)

if __name__ == '__main__':
    timestep_split_len = 2
    from multiprocessing import pool
    p = pool.Pool(processes=5)
    for label_location in [timestep_len + 2, timestep_len + 4, timestep_len + 6, timestep_len + 8, timestep_len + 10]:
        p.apply_async(main, args=(label_location, timestep_split_len, ))
    p.close()
    p.join()

    # 单任务
    # label_location = timestep_len + 2
    # main(label_location, timestep_split_len)
