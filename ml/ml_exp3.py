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
from sklearn.metrics import mean_squared_error, mean_absolute_error

'''
时间序列金额预测baseline
'''

# from kerascode.NNUtils import *
sys.path.append('/home/liwenjie/liwenjie/projects/lwjpaper')
from kerascode.configure import *
from kerascode.NNoperator import run_model
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
xlist = xlist.reshape(xlist.shape[0], -1)
xlist = np.concatenate((xlist, currlist), axis=1)
ylist, indexer = pd.factorize(ylist.reshape(-1))
x_train, x_test, y_train, y_test = train_test_split(xlist, ylist, test_size=0.2, random_state=42, stratify=ylist)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')


def exp_gbdt():
    train_data = lightgbm.Dataset(x_train, label=y_train, categorical_feature=[i for i in range(x_train.shape[1])])
    test_data = lightgbm.Dataset(x_test, label=y_test)
    parameters = {
        'objective': 'regression',
        'metric': 'l2',
        'boosting': 'gbdt',
        # 'num_leaves': 5,
        # 'feature_fraction': 0.5,
        # 'bagging_fraction': 0.5,
        # 'bagging_freq': 20,
        'learning_rate': 0.05,
        'verbose': 1,
        # 'min_data': 1,
    }

    model = lightgbm.train(parameters,
                           train_data,
                           valid_sets=test_data,
                           num_boost_round=2000,
                           early_stopping_rounds=100)

    yp_prob = model.predict(x_test)
    print('GBDT MSE: ', mean_squared_error(y_test, yp_prob))
    print('GBDT MAE: ', mean_absolute_error(y_test, yp_prob))


print('task 2: lightGBM ', confs)
exp_gbdt()


'''
金额预测
GBDT MSE:  13.236199464397597
GBDT MAE:  2.3967146225655522

Elastic Net MSE:  15.799247170735068
Elastic Net MAE:  2.8381716056849795
'''
