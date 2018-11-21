#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'liwenjie'

import os
import sys
from dateutil.parser import parse
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')

consum = pd.read_csv('../data/consumption.txt')
consum.drop(['id'], axis=1,inplace=True)
consum.fillna('',inplace=True)
consum['brush_time'] = pd.to_datetime(consum.brush_time)
consum['timeslot'] = '0'
# consum.index = consum.brush_time
consum = consum[consum.student_id != '']

# '''
# sleep 0:0 6:45
# breakfast 6:45 8:30
# morning 8:30 11:55
# noon 11:55 14:30
# afternoon 14:30 17:55
# dinner 17:55
# night 19:30
# sleep 21:55
# '''
# consum.loc[consum.between_time('0:0', '6:45', include_end=False).index, 'timeslot'] = 'sleep'
# consum.loc[consum.between_time('6:45', '8:30', include_end=False).index, 'timeslot'] = 'breakfast'
# consum.loc[consum.between_time('8:30', '11:55', include_end=False).index, 'timeslot'] = 'morning'
# consum.loc[consum.between_time('11:55', '14:30', include_end=False).index, 'timeslot'] = 'noon'
# consum.loc[consum.between_time('14:30', '17:55', include_end=False).index, 'timeslot'] = 'afternoon'
# consum.loc[consum.between_time('17:55', '19:30', include_end=False).index, 'timeslot'] = 'dinner'
# consum.loc[consum.between_time('19:30', '21:55', include_end=False).index, 'timeslot'] = 'night'
# consum.loc[consum.between_time('21:55', '0:0', include_end=False).index, 'timeslot'] = 'sleep'
#
# consum=consum.reset_index(drop=True)
# consum.drop(['brush_time'], axis=1, inplace=True)

import re
'''
null

# 沙河校区_ 沙河图书馆 只有1类

'''
pattern = [
    '电子科技大学_(\w{3})\dF.+',
    'null_(\w+)',
    '沙河校区_(\w+)',
    '电子科技大学_(.+)POS.+',
    '电子科技大学_(.+)收费.+',
    '(.+)_.+'  # general
]
def splitplace(x):
    res = x
    for p in pattern:
        if re.match(p, x):
            res = re.findall(p, x)[0]
            return res
    if x:
        print(x)
    return res
consum['place0'] = consum.place.apply(splitplace)
t, indexer = pd.factorize(consum.place0)
consum['placei'] = t
def label_next(x):
    try:
        if x.shape[0] == 1:
            label = [np.nan]
        else:
            label = [np.nan] + x['placei'][:-1].tolist()
        x['place_next'] = label
        return x
    except Exception as e:
        print(x)
res = consum.groupby(['student_id']).apply(label_next)

# consum.to_csv('../data/consum_cleaned.csv', index=False)
# consum.iloc[:-2674186*2, :].to_csv('../data/tf_train.csv', index=False, header=False)
# consum.iloc[-2674186*2:-2674186, :].to_csv('../data/tf_val.csv', index=False, header=False)
# consum.iloc[-2674186:, :].to_csv('../data/tf_test.csv', index=False, header=False)