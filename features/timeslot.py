#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'liwenjie'

import os
import sys
from dateutil.parser import parse
import pandas as pd
import numpy as np
import datetime
import json
# import matplotlib.pyplot as plt
# import seaborn as sns
from sqlalchemy import create_engine
import re
# plt.style.use('fivethirtyeight')
import warnings

warnings.filterwarnings('ignore')


def traindata1():
    '''
    训练数据第一版
    :return: 
    '''
    consum = pd.read_csv('../data/consumption.txt')
    consum.drop(['id'], axis=1, inplace=True)
    consum.dropna(subset=['place', 'student_id', 'brush_time'], inplace=True)
    consum.fillna('', inplace=True)
    consum['brush_time'] = pd.to_datetime(consum.brush_time)
    consum['timeslot'] = '0'
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
    consum.index = consum.brush_time
    '''
    sleep 0:0 6:45
    breakfast 6:45 8:30
    morning 8:30 11:55
    noon 11:55 14:30
    afternoon 14:30 17:55
    dinner 17:55
    night 19:30
    sleep 21:55
    '''
    consum.loc[consum.between_time('0:0', '6:45', include_end=False).index, 'timeslot'] = 'sleep'
    consum.loc[consum.between_time('6:45', '8:30', include_end=False).index, 'timeslot'] = 'breakfast'
    consum.loc[consum.between_time('8:30', '11:55', include_end=False).index, 'timeslot'] = 'morning'
    consum.loc[consum.between_time('11:55', '14:30', include_end=False).index, 'timeslot'] = 'noon'
    consum.loc[consum.between_time('14:30', '17:55', include_end=False).index, 'timeslot'] = 'afternoon'
    consum.loc[consum.between_time('17:55', '19:30', include_end=False).index, 'timeslot'] = 'dinner'
    consum.loc[consum.between_time('19:30', '21:55', include_end=False).index, 'timeslot'] = 'night'
    consum.loc[consum.between_time('21:55', '0:0', include_end=False).index, 'timeslot'] = 'sleep'
    consum = consum.reset_index(drop=True)
    consum.drop(['campus', 'place', 'place0', 'device_id', 'device_name', 'brush_time'], axis=1, inplace=True)

    eval_size = int(consum.shape[0] * 0.15)
    consum.to_csv('../data/consum_cleaned.csv', index=False)
    consum.iloc[:-eval_size * 2, :].to_csv('../data/tf_train.csv', index=False, header=False)
    consum.iloc[-eval_size * 2:-eval_size, :].to_csv('../data/tf_val.csv', index=False, header=False)
    consum.iloc[-eval_size:, :].to_csv('../data/tf_test.csv', index=False, header=False)


def traindata2():
    '''
    训练数据第二版：将标签变为下次刷卡
    :return: 
    '''
    consum = pd.read_csv('../data/consumption.txt')
    consum.drop(['id'], axis=1, inplace=True)
    consum.dropna(subset=['student_id', 'brush_time'], inplace=True)
    consum.fillna('', inplace=True)
    consum['brush_time'] = pd.to_datetime(consum.brush_time)
    consum['timeslot'] = '0'
    consum = consum.sort_values(['brush_time'])
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
        if x.shape[0] == 1:
            label = [np.nan]
        else:
            # error: label last
            # label = [np.nan] + x['placei'][:-1].tolist()
            label = x['placei'][1:].tolist() + [np.nan]
        x['place_next'] = label
        return x

    consum = consum.groupby(['student_id']).apply(label_next)
    consum = consum.dropna(subset=['place_next'])
    consum['place_next'] = consum['place_next'].apply(int)
    consum.index = consum.brush_time
    '''
    sleep 0:0 6:45
    breakfast 6:45 8:30
    morning 8:30 11:55
    noon 11:55 14:30
    afternoon 14:30 17:55
    dinner 17:55
    night 19:30
    sleep 21:55
    '''
    consum.loc[consum.between_time('0:0', '6:45', include_end=False).index, 'timeslot'] = 'sleep'
    consum.loc[consum.between_time('6:45', '8:30', include_end=False).index, 'timeslot'] = 'breakfast'
    consum.loc[consum.between_time('8:30', '11:55', include_end=False).index, 'timeslot'] = 'morning'
    consum.loc[consum.between_time('11:55', '14:30', include_end=False).index, 'timeslot'] = 'noon'
    consum.loc[consum.between_time('14:30', '17:55', include_end=False).index, 'timeslot'] = 'afternoon'
    consum.loc[consum.between_time('17:55', '19:30', include_end=False).index, 'timeslot'] = 'dinner'
    consum.loc[consum.between_time('19:30', '21:55', include_end=False).index, 'timeslot'] = 'night'
    consum.loc[consum.between_time('21:55', '0:0', include_end=False).index, 'timeslot'] = 'sleep'
    consum = consum.reset_index(drop=True)

    consum.drop(['campus', 'place', 'place0', 'device_id', 'device_name', 'brush_time', 'placei'], axis=1, inplace=True)
    consum.dropna(subset=['place_next'], inplace=True)
    consum.to_csv('../data/consum_cleaned_label_next.csv', index=False)
    eval_size = int(consum.shape[0] * 0.15)
    consum.iloc[:-eval_size * 2, :].to_csv('../data/tf_train_label_next.csv', index=False, header=False)
    consum.iloc[-eval_size * 2:-eval_size, :].to_csv('../data/tf_val_label_next.csv', index=False, header=False)
    consum.iloc[-eval_size:, :].to_csv('../data/tf_test_label_next.csv', index=False, header=False)


def traindata3():
    '''
    数据按学号分组
    :return: 
    '''
    consum = pd.read_csv('../data/consum_cleaned.csv', nrows=None)

    # def manyrowstoone(x):
    #     x = x.apply(lambda row: ','.join(row.apply(str)), axis=1)
    #     return ';'.join(x.apply(str))
    #
    # consum_timestep = consum.groupby(['student_id']).apply(manyrowstoone)
    # consum_timestep.to_csv('../data/consum_grouped.csv', index=False)

    dic_feat = {}
    for col in consum.columns:
        dic_feat[col] = []

    def grp(x):
        for col in x.columns:
            dic_feat[col].append(x[col].values)

    consum.groupby(['student_id']).apply(grp)
    consum['card_id'] = consum['card_id'].apply(str)
    consum['card_id'] = consum['card_id'].apply(str)

    def pad_to_dense(M, fill_value, dtype=np.float32):
        """Appends the minimal required amount of zeroes at the end of each 
         array in the jagged array `M`, such that `M` looses its jagedness."""

        maxlen = max(len(r) for r in M)

        Z = np.full(shape=(len(M), maxlen), fill_value=fill_value, dtype=dtype)
        for enu, row in enumerate(M):
            Z[enu, :len(row)] += row
        return Z

    for col in dic_feat.keys():
        if isinstance(dic_feat[col][0][0], str):
            r = pad_to_dense(dic_feat[col], fill_value='', dtype=object)
        else:
            r = pad_to_dense(dic_feat[col], fill_value=0.0)
        dic_feat[col] = r
    for k in dic_feat.keys():
        np.save('../data/stutimestep_%s.npy'%k, dic_feat[k])


def traindata4():
    '''
    训练数据,按学号分组，学号转换为int
    :return: 
    '''
    consum = pd.read_csv('../data/consumption.txt')
    consum.drop(['id'], axis=1, inplace=True)
    consum.dropna(subset=['place', 'student_id', 'brush_time'], inplace=True)
    consum.fillna('', inplace=True)
    consum['brush_time'] = pd.to_datetime(consum.brush_time)
    consum['timeslot'] = '0'
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
    consum.index = consum.brush_time
    '''
    sleep 0:0 6:45
    breakfast 6:45 8:30
    morning 8:30 11:55
    noon 11:55 14:30
    afternoon 14:30 17:55
    dinner 17:55
    night 19:30
    sleep 21:55
    '''
    consum.loc[consum.between_time('0:0', '6:45', include_end=False).index, 'timeslot'] = 'sleep'
    consum.loc[consum.between_time('6:45', '8:30', include_end=False).index, 'timeslot'] = 'breakfast'
    consum.loc[consum.between_time('8:30', '11:55', include_end=False).index, 'timeslot'] = 'morning'
    consum.loc[consum.between_time('11:55', '14:30', include_end=False).index, 'timeslot'] = 'noon'
    consum.loc[consum.between_time('14:30', '17:55', include_end=False).index, 'timeslot'] = 'afternoon'
    consum.loc[consum.between_time('17:55', '19:30', include_end=False).index, 'timeslot'] = 'dinner'
    consum.loc[consum.between_time('19:30', '21:55', include_end=False).index, 'timeslot'] = 'night'
    consum.loc[consum.between_time('21:55', '0:0', include_end=False).index, 'timeslot'] = 'sleep'
    consum = consum.reset_index(drop=True)
    consum.drop(['campus', 'place', 'place0', 'device_id', 'device_name', 'brush_time'], axis=1, inplace=True)

    stuidlabels, stuiduniques = pd.factorize(consum['student_id'])
    stuidmap = {sid:i for i, sid in enumerate(stuiduniques)}
    json.dump(stuidmap, open('../data/stuidmap.json', 'w'))
    consum['student_id'] = stuidlabels

    eval_size = int(consum.shape[0] * 0.15)
    consum.to_csv('../data/consum_cleaned_groupby.csv', index=False)
    consum.iloc[:-eval_size * 2, :].to_csv('../data/tf_train_groupby.csv', index=False)
    consum.iloc[-eval_size * 2:-eval_size, :].to_csv('../data/tf_val_groupby.csv', index=False)
    consum.iloc[-eval_size:, :].to_csv('../data/tf_test_groupby.csv', index=False)


def traindata5():
    '''
    训练数据,ML使用，将时间每半个小时分为一个bin，每天分为48个bin，同时将消费和门禁合并
    :return: 
    '''
    DB_OUT_HOST = '172.16.170.81'
    DB_OUT_PORT = 3307
    DB_OUT_USER = 'liwenjie'
    DB_OUT_PWD = 'liwenjie333'
    DB_OUT_NAME = 'data_center'

    engine = create_engine(
        'mysql+pymysql://%(user)s:%(pwd)s@%(host)s:%(port)s/%(dbname)s?charset=utf8' % {'user': DB_OUT_USER,
                                                                                        'pwd': DB_OUT_PWD,
                                                                                        'host': DB_OUT_HOST,
                                                                                        'dbname': DB_OUT_NAME,
                                                                                        'port': DB_OUT_PORT},
        encoding='utf-8')
    conn = engine.connect()
    # 基本信息
    df_basicinfo = pd.read_sql_table('t_basicinfo', conn)

    consum = pd.read_csv('../data/consumption.txt')
    consum.drop(['id'], axis=1, inplace=True)
    consum.dropna(subset=['place', 'student_id', 'brush_time'], inplace=True)
    consum.fillna('', inplace=True)
    consum['brush_time'] = pd.to_datetime(consum.brush_time)
    consum['timeslot'] = '0'
    consum = consum[consum['student_id'].isin(df_basicinfo['student_id'])]

    df_access = pd.read_csv('../data/access.txt')
    df_access = df_access.drop(['campus', 'device_id', 'id'], axis=1)
    df_access = df_access[df_access['student_id'].isin(df_basicinfo['student_id'])]
    df_access = df_access.rename({'pass_time':'brush_time','direction':'trans_type'}, axis=1)

    consum = pd.concat([consum, df_access])
    consum = consum


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

    # 添加下次刷卡地点标签
    def label_next(x):
        if x.shape[0] == 1:
            label = [np.nan]
        else:
            # error: label last
            # label = [np.nan] + x['placei'][:-1].tolist()
            label = x['placei'][1:].tolist() + [np.nan]
        x['place_next'] = label
        return x

    consum = consum.groupby(['student_id']).apply(label_next)
    consum = consum.dropna(subset=['place_next'])
    consum['place_next'] = consum['place_next'].apply(int)

    consum.index = consum.brush_time
    '''
    30分钟一个band
    '''
    binband = pd.date_range(start='2019-01-01', end='2019-01-02', freq='30Min')
    for i in range(binband.shape[0]-1):
        stime = binband[i].strftime('%H:%M:%S')
        etime = binband[i+1].strftime('%H:%M:%S')
        consum.loc[consum.between_time(stime, etime, include_end=False).index, 'timeslot'] = i
    consum = consum.reset_index(drop=True)
    consum.drop(['campus', 'device_id', 'device_name'], axis=1, inplace=True)

    # stuidlabels, stuiduniques = pd.factorize(consum['student_id'])
    # stuidmap = {sid:i for i, sid in enumerate(stuiduniques)}
    # json.dump(stuidmap, open('../data/stuidmap.json', 'w'))
    # consum['student_id'] = stuidlabels

    eval_size = int(consum.shape[0] * 0.15)
    consum.to_csv('../data/consum_access_feat5.csv', index=False)
    consum.iloc[:-eval_size * 2, :].to_csv('../data/tf_train_feat5.csv', index=False)
    consum.iloc[-eval_size * 2:-eval_size, :].to_csv('../data/tf_val_feat5.csv', index=False)
    consum.iloc[-eval_size:, :].to_csv('../data/tf_test_feat5.csv', index=False)

if __name__ == '__main__':
    traindata1()
