#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'lkjie'

import os
import sys
from dateutil.parser import parse
import pandas as pd
import numpy as np
import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import re
from sqlalchemy import create_engine
plt.style.use('fivethirtyeight')
import warnings
import lightgbm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
sns.set(font='SimHei')
import logging
logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(threadName)s -  %(levelname)s - %(message)s')
logging.info('start...')


# 一个学期的数据，从20150901-20160117，学生包括4个年级的所有本科生，已补全方向，地点重新整理保留164个不同地点
# consum.to_csv('../data/consum_access_feat54.csv', index=False)
consum = pd.read_csv('../data/consum_access_feat54.csv', nrows=200000)
consum['brush_time'] = pd.to_datetime(consum['brush_time'])

features = ['student_id_int', 'timeslot']
label = 'placei'
label_cates = consum[label].drop_duplicates().count()
print('label_cates: %d'%label_cates)

x, x_test, y, y_test = train_test_split(consum[features], consum[label], test_size=0.2, random_state=42)
categorical_features = ['student_id_int', 'timeslot']
train_data = lightgbm.Dataset(x, label=y, categorical_feature=categorical_features)
test_data = lightgbm.Dataset(x_test, label=y_test)
parameters = {
    'objective': 'multiclass',
    "num_class": label_cates,
    'metric': 'multi_logloss',
    'is_unbalance': 'true',
    'boosting': 'gbdt',
    'num_leaves': 5,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 20,
    'learning_rate': 0.05,
    'verbose': -1,
    'min_data': 1,
}

model = lightgbm.train(parameters,
                       train_data,
                       valid_sets=test_data,
                       num_boost_round=5,
                       early_stopping_rounds=100)

yp_prob = model.predict(x_test)
yp = yp_prob.argmax(axis=1)

k = 3
predictions = tf.Variable(yp_prob, dtype=tf.float32)
targets = tf.Variable(y_test)
topk = tf.nn.in_top_k(
    predictions,
    targets,
    k,
    name=None
)

init_op = tf.global_variables_initializer()
config = tf.ConfigProto(device_count={"CPU": 1, 'GPU': 0})
tfsess = tf.Session(config=config)
tfsess.run(init_op)


def topk_tf(ytrue, ypred, k=1):
    assert len(ytrue.shape) == 1 and len(ypred.shape) == 2
    predictions = tf.Variable(ypred, dtype=tf.float32)
    targets = tf.Variable(ytrue)
    topk = tf.nn.in_top_k(
        predictions,
        targets,
        k,
        name=None
    )
    tfsess.run(predictions.initializer)
    tfsess.run(targets.initializer)
    res = tfsess.run(topk)
    return res.sum() / res.shape[0]


top3 = topk_tf(y_test, yp_prob, k=3)
top5 = topk_tf(y_test, yp_prob, k=5)

print(top3, top5)
