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


#importing all the required ML packages
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn import svm #support vector Machine
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #Naive bayes
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix

# 一个学期的数据，从20150901-20160117，学生包括4个年级的所有本科生，已补全方向，地点重新整理保留164个不同地点，添加timeslot_week
# consum.to_csv('../data/consum_access_feat54.csv', index=False)
consum = pd.read_csv('../data/consum_access_feat55.csv')
consum['brush_time'] = pd.to_datetime(consum['brush_time'])

config = tf.ConfigProto(device_count={"CPU": 1, 'GPU': 0})
tfsess = tf.InteractiveSession(config=config)

def mertics_acck(y_true, y_pred):
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

    init_op = tf.global_variables_initializer()
    tfsess.run(init_op)
    top1 = topk_tf(y_true, y_pred, k=1)
    top3 = topk_tf(y_true, y_pred, k=3)
    top5 = topk_tf(y_true, y_pred, k=5)
    top10 = topk_tf(y_true, y_pred, k=10)
    print('top1, top3, top5, top10')
    print(top1, top3, top5, top10)


import numpy as np
import pandas as pd
import lightgbm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

'''GDBT 实验三'''
consum = pd.read_csv('../data/consum_access_feat54.csv', nrows=200000)
consum['brush_time'] = pd.to_datetime(consum['brush_time'])

features = ['student_id_int', 'timeslot']
label = 'placei'
label_cates = consum[label].drop_duplicates().count()
print('label_cates: %d'%label_cates)

x_train, x_test, y_train, y_test = train_test_split(consum[features], consum[label], test_size=0.2, random_state=42)
categorical_features = ['student_id_int', 'timeslot']


def exp_gbdt():
    train_data = lightgbm.Dataset(x_train, label=y_train, categorical_feature=categorical_features)
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
        'verbose': 0,
        'min_data': 1,
    }

    model = lightgbm.train(parameters,
                           train_data,
                           valid_sets=test_data,
                           num_boost_round=500,
                           early_stopping_rounds=100)

    yp_prob = model.predict(x_test)

    mertics_acck(y_test, yp_prob)


from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
x = enc.fit_transform(consum[features])
y = consum[label]
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(x_train.shape[0], 'train sequences')
print(x_test.shape[0], 'test sequences')


def exp_logistic():
    model = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    model.fit(x_train,y_train)
    prediction1=model.predict_proba(x_test)
    mertics_acck(y_test, prediction1)


def exp_svm():
    model = svm.SVC(kernel='rbf', C=1, gamma=0.1)
    model.fit(x_train,y_train)
    prediction1=model.predict_proba(x_test)
    mertics_acck(y_test, prediction1)


def exp_bayes():
    from sklearn.naive_bayes import GaussianNB  # Naive bayes
