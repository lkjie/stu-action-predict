#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'lkjie'

import pandas as pd
import os
import numpy as np
from joblib import Parallel, delayed
import multiprocessing

'''
配置文件，不允许引用其他内部包，防止交叉引用
所有神经网络均使用如下配置
'''

# 是否取全量数据
alldata = True
# 如果不为全量数据，取前多少行
nrows = 500000
# 神经网络batch大小
batch_size = 32
# 时间序列长度
timestep_len = 2
# 时间序列分割步长，默认为timestep_len+1
timestep_split_len = timestep_len + 1
# 特征分割是否分层
stratify = True
# 神经网络迭代轮数
epochs = 40

PROJECT_DIR = '/home/liwenjie/liwenjie/projects/lwjpaper'
confs = 'batchsize%d_maxlen%d' % (batch_size, timestep_len)


if not os.path.exists('logs'):
    os.makedirs('logs')
if not os.path.exists('models'):
    os.makedirs('models')
if not os.path.exists('array'):
    os.makedirs('array')


def get_experiment_name(name):
    if alldata:
        return '%s_alldata_%s' % (name, confs)
    else:
        return '%s_%drows_%s' % (name, nrows, confs)


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


def gen_data_exptl(timeseries, labels, expid):
    '''
    exp 8 12
    :return:
    '''

    # 更改数据
    def to_seq(x):
        xlist = []
        ylist = []
        for i in range(0, x.shape[0], timestep_split_len):
            try:
                if x.shape[0] < i + timestep_split_len:
                    break
                dfx = x.iloc[i:i + timestep_len][timeseries]
                dfy = x.iloc[i + timestep_len][labels]
                xlist.append(dfx.values)
                ylist.append(dfy.values)
            except Exception as e:
                print(e)
        if xlist:
            return xlist, ylist
        else:
            return None

    # consum.groupby(['student_id_int']).apply(to_seq)
    res = applyParallel(consum.groupby(['student_id_int']), to_seq)

    xlist = ''
    ylist = ''
    for ele in res:
        if ele:
            if isinstance(xlist, str):
                xlist = np.array(ele[0])
                ylist = np.array(ele[1])
            else:
                x = np.array(ele[0])
                xlist = np.concatenate((xlist, x))
                y = np.array(ele[1])
                ylist = np.concatenate((ylist, y))
    np.save('array/exp%d_xlist_%s.npy' % (expid, confs), xlist)
    np.save('array/exp%d_ylist_%s.npy' % (expid, confs), ylist)
    return xlist, ylist


def gen_data_expftl(features, timeseries, labels, expid):
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
                if x.shape[0] < i + timestep_split_len:
                    return
                dfx = x.iloc[i:i + timestep_len][timeseries]
                curr = x.iloc[i + timestep_len][features]
                dfy = x.iloc[i + timestep_len][labels]
                xlist.append(dfx.values)
                currlist.append(curr.values)
                ylist.append(dfy.values)
            except Exception as e:
                print(e)
        if xlist:
            return xlist, currlist, ylist
        else:
            return None

    res = applyParallel(consum.groupby(['student_id_int']), to_seq)
    # consum.groupby(['student_id_int']).apply(to_seq)

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
    np.save('array/exp%d_currlist_%s.npy' % (expid, confs), currlist)
    np.save('array/exp%d_xlist_%s.npy' % (expid, confs), xlist)
    np.save('array/exp%d_ylist_%s.npy' % (expid, confs), ylist)
    return xlist, currlist, ylist


def load_data_expftl(features, timeseries, labels, expid):
    if os.path.exists('array/exp%d_xlist_%s.npy' % (expid, confs)):
        xlist = np.load('array/exp%d_xlist_%s.npy' % (expid, confs))
        currlist = np.load('array/exp%d_currlist_%s.npy' % (expid, confs))
        ylist = np.load('array/exp%d_ylist_%s.npy' % (expid, confs))
        return xlist, currlist, ylist
    else:
        return gen_data_expftl(features, timeseries, labels, expid)


def load_data_exptl(timeseries, labels, expid):
    if os.path.exists('array/exp%d_xlist_%s.npy' % (expid, confs)):
        xlist = np.load('array/exp%d_xlist_%s.npy' % (expid, confs))
        ylist = np.load('array/exp%d_ylist_%s.npy' % (expid, confs))
        return xlist, ylist
    else:
        return gen_data_exptl(timeseries, labels, expid)
