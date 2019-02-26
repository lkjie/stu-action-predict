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


consum = pd.read_csv('../data/consum_access_feat6m.csv')

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


import pandas as pd
import lightgbm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


'''GDBT 实验三'''

features = ['student_id_int', 'timeslot_week']
labels = ['placei']
labels_cates = [consum[f].drop_duplicates().count() for f in labels]
print('label_cates: %d'%labels_cates[0])

x_train, x_test, y_train, y_test = train_test_split(consum[features], consum[labels], test_size=0.2, random_state=42, stratify=consum[labels])
categorical_features = ['student_id_int', 'timeslot_week']


def exp_gbdt():
    train_data = lightgbm.Dataset(x_train, label=y_train, categorical_feature=categorical_features)
    test_data = lightgbm.Dataset(x_test, label=y_test)
    parameters = {
        'objective': 'multiclass',
        "num_class": labels_cates[0],
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


def exp_bayes():
    from sklearn.naive_bayes import GaussianNB  # Naive bayes
    model=GaussianNB()
    model.fit(x_train, y_train)
    prediction6=model.predict_proba(x_test)
    mertics_acck(y_test, prediction6)


# print('task 1: lightGBM')
# exp_gbdt()

# print('task 1: naive_bayes')
# exp_bayes()

from sklearn.preprocessing import OneHotEncoder
import time
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

# enc = OneHotEncoder(handle_unknown='ignore')
# x = enc.fit_transform(consum[features])
# # Hash
# from sklearn.feature_extraction import FeatureHasher
#
# y = consum[labels]
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
# print(x_train.shape[0], 'train sequences')
# print(x_test.shape[0], 'test sequences')
# print(x_train.shape[1], ' features')

def exp_logistic():
    model = LogisticRegression(random_state=0, solver='sag', multi_class='multinomial', n_jobs=-1, verbose=1)
    model.fit(x_train,y_train)
    prediction1=model.predict_proba(x_test)
    mertics_acck(y_test, prediction1)


def exp_svm():
    model = svm.SVC(kernel='rbf', C=1, gamma=0.1, verbose=1, shrinking=False)
    model.fit(x_train,y_train)
    prediction1=model.predict_proba(x_test)
    mertics_acck(y_test, prediction1)


def baggin_svm():
    n_estimators = 100
    start = time.time()
    clf = OneVsRestClassifier(
        BaggingClassifier(SVC(kernel='rbf', probability=True, class_weight='balanced'), max_samples=1.0 / n_estimators,
                          n_estimators=n_estimators))
    clf.fit(x_train, y_train)
    end = time.time()
    print("Bagging SVC", end - start, clf.score(x_train, y_train))
    prediction1 = clf.predict_proba(x_test)
    mertics_acck(y_test, prediction1)


'''
LR 实验一 用6m, ['student_id_int', 'timeslot']时 0.4223056919283194 0.7125646434444081 0.822104074027294 0.9270232911171261
'''


'''
地点推荐 data：6m ['student_id_int', 'timeslot_week']
GBDT 0.35340973880566184 0.5901561358300271 0.7045152730281757 0.8362716270021614 (epoch=2000)
NB 0.31751042026944554 0.4820473896502024 0.5842634159124686 0.7428687110973656
LR 0.4261302937978887 0.7164388234863607 0.8252912422519351 0.9280856805253398
'''


# print('task 1: lr')
# exp_logistic()

# print('task 1: svm')
# exp_svm() # too slow

# print('task 1: baggin_svm')
# baggin_svm()


