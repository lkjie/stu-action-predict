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
nrows = 2000000
# 神经网络batch大小
batch_size = 32
# 时间序列长度
timestep_len = 10

# 特征分割是否分层
stratify = True
# 神经网络迭代轮数
epochs = 40

PROJECT_DIR = '/home/liwenjie/liwenjie/projects/lwjpaper'
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


def gen_data_exptl(timeseries, labels, expid, df=consum, label_location=timestep_len + 1, timestep_split_len=timestep_len + 1):
    '''
    exp 8 12
    # label_location : 标签位置，默认是时间序列后面的一个timestep_len + 1，为了预测不止下一个时间，可以更改
    # timestep_split_len: 时间序列分割步长，默认为timestep_len+1
    :return:
    '''

    # 更改数据
    def to_seq(x):
        xlist = []
        ylist = []
        for i in range(0, x.shape[0], timestep_split_len):
            try:
                if x.shape[0] <= i + label_location - 1:
                    break
                # 每个序列的长度为
                dfx = x.iloc[i:i + timestep_len][timeseries]
                dfy = x.iloc[i + label_location - 1][labels]
                xlist.append(dfx.values)
                ylist.append(dfy.values)
            except Exception as e:
                print(e)
        if xlist:
            return xlist, ylist
        else:
            return None

    # df.groupby(['student_id_int']).apply(to_seq)
    res = applyParallel(df.groupby(['student_id_int']), to_seq)

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
    np.save('array/exp%d_xlist_%s_%d_%d.npy' % (expid, confs, label_location, timestep_split_len), xlist)
    np.save('array/exp%d_ylist_%s_%d_%d.npy' % (expid, confs, label_location, timestep_split_len), ylist)
    return xlist, ylist


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


def load_data_expftl(features, timeseries, labels, expid, label_location=timestep_len + 1, timestep_split_len=timestep_len + 1):
    if os.path.exists('array/exp%d_xlist_%s_%d_%d.npy' % (expid, confs, label_location, timestep_split_len)):
        xlist = np.load('array/exp%d_xlist_%s_%d_%d.npy' % (expid, confs, label_location, timestep_split_len))
        currlist = np.load('array/exp%d_currlist_%s_%d_%d.npy' % (expid, confs, label_location, timestep_split_len))
        ylist = np.load('array/exp%d_ylist_%s_%d_%d.npy' % (expid, confs, label_location, timestep_split_len))
        return xlist, currlist, ylist
    else:
        return gen_data_expftl(features, timeseries, labels, expid, label_location=label_location, timestep_split_len=timestep_split_len)


def load_data_exptl(timeseries, labels, expid, label_location=timestep_len + 1, timestep_split_len=timestep_len + 1):
    if os.path.exists('array/exp%d_xlist_%s_%d_%d.npy' % (expid, confs, label_location, timestep_split_len)):
        xlist = np.load('array/exp%d_xlist_%s_%d_%d.npy' % (expid, confs, label_location, timestep_split_len))
        ylist = np.load('array/exp%d_ylist_%s_%d_%d.npy' % (expid, confs, label_location, timestep_split_len))
        return xlist, ylist
    else:
        return gen_data_exptl(timeseries, labels, expid, label_location=label_location, timestep_split_len=timestep_split_len)


def load_data_expftl_train_test(features, timeseries, labels, expid, label_location=timestep_len + 1, timestep_split_len=timestep_len + 1):
    '''
        exp 17
    由于实验17验证集的准确率高于训练集，可能的原因是在训练集与测试集切分时有leak，重新在数据生成阶段设计训练集与测试集
    :return:
    '''
    if os.path.exists('array/exp%d_x_train1_%s_%d_%d.npy' % (expid, confs, label_location, timestep_split_len)):
        xtrain1 = np.load('array/exp%d_xtrain1_%s_%d_%d.npy' % (expid, confs, label_location, timestep_split_len))
        xtest1 = np.load('array/exp%d_xtest1_%s_%d_%d.npy' % (expid, confs, label_location, timestep_split_len))
        xtrain2 = np.load('array/exp%d_xtrain2_%s_%d_%d.npy' % (expid, confs, label_location, timestep_split_len))
        xtest2 = np.load('array/exp%d_xtest2_%s_%d_%d.npy' % (expid, confs, label_location, timestep_split_len))
        ytrain = np.load('array/exp%d_ytrain_%s_%d_%d.npy' % (expid, confs, label_location, timestep_split_len))
        ytest = np.load('array/exp%d_ytest_%s_%d_%d.npy' % (expid, confs, label_location, timestep_split_len))
        return xtrain1, xtest1, xtrain2, xtest2, ytrain, ytest
    else:
        consum_train = pd.read_csv(PROJECT_DIR + '/data/consum_access_feat6m_train.csv')
        consum_test = pd.read_csv(PROJECT_DIR + '/data/consum_access_feat6m_test.csv')
        xtrain1, xtrain2, ytrain = gen_data_expftl(features, timeseries, labels, expid, df=consum_train, label_location=label_location, timestep_split_len=timestep_split_len)
        xtest1, xtest2, ytest = gen_data_expftl(features, timeseries, labels, expid, df=consum_test, label_location=label_location, timestep_split_len=timestep_split_len)
        return xtrain1, xtest1, xtrain2, xtest2, ytrain, ytest
