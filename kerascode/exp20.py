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

PROJECT_DIR = "/mnt/sdb/liwenjie/myProjects/lwjpaper"
# PROJECT_DIR = "/home/liwenjie/liwenjie/myProjects/lwjpaper"
sys.path.append(PROJECT_DIR)
from kerascode.NNUtils import *
from kerascode.NNoperator import run_model

'''
预测地点exp9，冷启动结果，分别针对1,2,3,4周的准确率进行实验，
其中：
第一周数据量 203641；
前二周数据量 423648；
前三周数据量 656611；
前四周数据量 869470；
'''

from joblib import Parallel, delayed
import multiprocessing

# 是否取全量数据
alldata = False
# 如果不为全量数据，取前多少行
nrows = 869470
# 神经网络batch大小
batch_size = 32
# 时间序列长度
timestep_len = 10

# 特征分割是否分层
stratify = True
# 神经网络迭代轮数
epochs = 40


if alldata:
    confs = 'alldata_batchsize%d_maxlen%d' % (batch_size, timestep_len)
else:
    confs = '%drows_batchsize%d_maxlen%d' % (nrows, batch_size, timestep_len)


if not os.path.exists('logs'):
    os.makedirs('logs')
if not os.path.exists('models'):
    os.makedirs('models')
if not os.path.exists('array'):
    os.makedirs('array')


def get_experiment_name(name):
    return '%s_%s' % (name, confs)


def load_data():
    if alldata:
        consum = pd.read_csv(PROJECT_DIR + '/data/consum_access_feat6m.csv')
    else:
        consum = pd.read_csv(PROJECT_DIR + '/data/consum_access_feat6m.csv', nrows=nrows)
    # consum['brush_time'] = pd.to_datetime(consum['brush_time'])
    return consum


consum = load_data()


def applyParallel(dfGrouped, func):
    retLst = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group) for name, group in dfGrouped)
    return retLst


def gen_data_expftl(features, timeseries, labels, expid, df=consum, label_location=timestep_len + 1, timestep_split_len=timestep_len + 1):
    '''
    exp 7 9 10
    :return:
    '''

    # 更改数据
    def to_seq(x):
        xlist = []
        currlist = []
        ylist = []
        for i in range(0, x.shape[0], timestep_split_len):
            try:
                if x.shape[0] <= i + label_location - 1:
                    break
                dfx = x.iloc[i:i + timestep_len][timeseries]
                curr = x.iloc[i + label_location - 1][features]
                dfy = x.iloc[i + label_location - 1][labels]
                xlist.append(dfx.values)
                currlist.append(curr.values)
                ylist.append(dfy.values)
            except Exception as e:
                print(e)
        if xlist:
            return xlist, currlist, ylist
        else:
            return None

    res = applyParallel(df.groupby(['student_id_int']), to_seq)
    # df.groupby(['student_id_int']).apply(to_seq)

    xlist = ''
    ylist = ''
    currlist = ''
    for ele in res:
        if ele:
            x, curr, y = ele
            x = np.array(x)
            curr = np.array(curr)
            if isinstance(xlist, str):
                xlist = x
                currlist = curr
                ylist = y
            else:
                xlist = np.concatenate((xlist, x))
                currlist = np.concatenate((currlist, curr))
                ylist = np.concatenate((ylist, y))
    np.save('array/exp%d_currlist_%s_%d_%d.npy' % (expid, confs, label_location, timestep_split_len), currlist)
    np.save('array/exp%d_xlist_%s_%d_%d.npy' % (expid, confs, label_location, timestep_split_len), xlist)
    np.save('array/exp%d_ylist_%s_%d_%d.npy' % (expid, confs, label_location, timestep_split_len), ylist)
    return xlist, currlist, ylist


experiment = os.path.basename(__file__).replace('.py', '')
experiment = get_experiment_name(experiment)

print('Loading data...')

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

xlist, currlist, ylist = gen_data_expftl(features, timeseries, labels, 9)
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
    timeseries_inp = Input(shape=(timestep_len, timeseries_count), dtype='int32')
    branch_outputs = []
    for i in range(timeseries_count):
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

    lstm1 = GRU(256, dropout=0.2, recurrent_dropout=0.2)(timeseries_x)

    branch_outputs = []
    fea_inp = Input(shape=(feature_count,), dtype='int32')
    for i in range(feature_count):
        out = Lambda(lambda x: x[:, i])(fea_inp)
        if features[i] == 'student_id_int':
            nextlayer = Embedding(input_dim=emb_feat_cates[i], output_dim=12, mask_zero=False,
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


model = build_model()
run_model(experiment, model, [x_train1, x_train2], [y_train], [x_test1, x_test2], [y_test], batch_size=batch_size, epochs=epochs)
