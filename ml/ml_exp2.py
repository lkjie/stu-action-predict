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

from kerascode.NNUtils import *
from kerascode.configure import *

print('Loading data...')
consum = load_data()
features = ['timeslot_week',
            # 'amount',
            # 'remained_amount',
            # 'trans_type',
            # 'category'
            ]
timeseries = ['student_id_int', 'timeslot_week', 'placei']
label = 'placei'
feature_count = len(features)
timeseries_count = len(timeseries)
label_cates = consum[label].drop_duplicates().count()
emb_feat_cates = [consum[f].drop_duplicates().count() for f in features]
emb_feat_names = ['emb_feat_%s' % f for f in features]
emb_timeseries_cates = [consum[f].drop_duplicates().count() for f in timeseries]
emb_timeseries_names = ['emb_timeseries_%s' % f for f in timeseries]

xlist, currlist, ylist = load_data_exp7910(features, timeseries, label, 9)
if stratify:
    unique, counts = np.unique(ylist, return_counts=True)
    idy = np.isin(ylist, unique[counts > 1]).reshape(-1)
    ylist = ylist[idy]
    xlist = xlist[idy]
    currlist = currlist[idy]

xlist = xlist.reshape(xlist.shape[0], -1)
xlist = np.concatenate((xlist, currlist), axis=1)
ylist = ylist.reshape(-1)
x_train, x_test, y_train, y_test = train_test_split(xlist, ylist, test_size=0.2, random_state=42, stratify=ylist)
