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
logging.basicConfig(level=logging.DEBUG,
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

'''
时间序列地点预测baseline
'''

# from kerascode.NNUtils import *
sys.path.append('/home/liwenjie/liwenjie/projects/lwjpaper')
from kerascode.configure import *
from kerascode.NNoperator import run_model
print('Loading data...')

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
xlist, currlist, ylist = load_data_expftl(features, timeseries, labels, 9)

if stratify:
    unique, counts = np.unique(ylist, return_counts=True)
    idy = np.isin(ylist, unique[counts > 1]).reshape(-1)
    ylist = ylist[idy]
    xlist = xlist[idy]
    currlist = currlist[idy]

xlist = xlist.reshape(xlist.shape[0], -1)
xlist = np.concatenate((xlist, currlist), axis=1)
ylist, indexer = pd.factorize(ylist.reshape(-1))
x_train, x_test, y_train, y_test = train_test_split(xlist, ylist, test_size=0.2, random_state=42, stratify=ylist)


config = tf.ConfigProto(device_count={"CPU": 1, 'GPU': 0})
tfsess = tf.Session(config=config)
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


def exp_gbdt():
    train_data = lightgbm.Dataset(x_train, label=y_train, categorical_feature=[i for i in range(x_train.shape[1])])
    test_data = lightgbm.Dataset(x_test, label=y_test)
    parameters = {
        'objective': 'multiclass',
        "num_class": label_cates[0],
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
                           num_boost_round=2000,
                           early_stopping_rounds=100)

    yp_prob = model.predict(x_test)

    mertics_acck(y_test, yp_prob)


# print('task 2: lightGBM ', confs)
# exp_gbdt()




def exp_bayes():
    from sklearn.naive_bayes import GaussianNB  # Naive bayes
    model=GaussianNB()
    model.fit(x_train, y_train)
    prediction6=model.predict_proba(x_test)
    mertics_acck(y_test, prediction6)


# print('task 2: naive_bayes', confs)
# exp_bayes()


from sklearn.preprocessing import OneHotEncoder
import time
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

enc = OneHotEncoder(handle_unknown='ignore')
x = enc.fit_transform(xlist)
x_train, x_test, y_train, y_test = train_test_split(x, ylist, test_size=0.2, random_state=42, stratify=ylist)


def exp_logistic():
    model = LogisticRegression(random_state=0, solver='sag', multi_class='multinomial', n_jobs=-1, verbose=1)
    model.fit(x_train, y_train)
    prediction1 = model.predict_proba(x_test)
    mertics_acck(y_test, prediction1)


print('task 2: lr', confs)
exp_logistic()

'''
下一次地点预测
top1, top3, top5, top10
GBDT 0.4640097600650671 0.7181781211874746 0.8049342551172564 0.895485969906466 (timestep_len=10)
GBDT 0.4763873687624144 0.7475779318172605 0.8354209736916778 0.917872633669788 (timestep_len=5)
GBDT 0.49822436857708546 0.7746886097696393 0.8595836956180141 0.9350192229124545 (timestep_len=2) loss 1.7667
NBC 0.13693299282500304 0.2839190887348494 0.40865053305768373 0.6348858891726459 (timestep_len=5)
NBC 0.30806943295027184 0.48040520013827925 0.5829623189013085 0.7384426821986403 (timestep_len=2)
LR 0.4966634890371783 0.7765532846562399 0.8644967996731581 0.9427921935071601 (timestep_len=2)
'''
