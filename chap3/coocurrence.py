#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'lkjie'

import pandas as pd
import pymysql, logging
import sys, datetime
from sqlalchemy import create_engine
import numpy as np
from dateutil.relativedelta import relativedelta
from scipy.stats import norm
import matplotlib.pyplot as plt
import queue, time, dateutil, math
from collections import namedtuple
import json
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_alg
from scipy.sparse import csgraph
import itertools
import os

from utils import util

# append work dir
workdir = os.path.dirname(os.path.dirname(__file__))
if workdir not in sys.path:
    sys.path.append(workdir)

'''
require: MySQLdb
input_database:
    t_basicinfo as b:
        b.student_id,
        b.name,
        b.gender,
        b.college,
        b.major,
        b.admission_grade
    t_consumption as c:
        c.student_id,
        c.brush_time,
        c.amount,
        c.place,
        c.category
output_database:
    t_friend:
        # all field

'''
LEAST_CO_TIMES = 6  # 最少共现次数，和数据量相符，此处利用六个月数据，次数为6
begin_year = 2012
stubasicinfo = util.get_basicinfo()
STU_ID_STD_LIST = stubasicinfo['student_id'].tolist()
STU_NAME_STD_LIST = {row['student_id']: row['name'] for index, row in stubasicinfo.iterrows()}
STU_NAME_LIST = stubasicinfo['name'].tolist()


class TimeWindow:
    timewindow = []

    def __init__(self, student_id_list):
        '''
        初始化
        :param stunums:学号列表
        :return:
        '''
        stulen = len(student_id_list)
        stuidlist = pd.Series(student_id_list)
        if stuidlist.empty:
            logging.error('TimeWindow Init: student id is NULL!')
        elif stuidlist.duplicated().sum() > 0:
            stuidlist = stuidlist.drop_duplicates()
            logging.warning('TimeWindow Init: student id has duplicated, already drop duplicates!')
        self.stuidlist = stuidlist.tolist()
        self.stuid_index_map = {k: i for i, k in enumerate(self.stuidlist)}
        self.cc_matrix = sparse.dok_matrix((stulen, stulen), dtype=np.int16)

    def __addOneRecord(self, stuid, stutime, location):
        '''

        :param stuid:
        :param stutime: must be datetime
        :param location:
        :return:
        '''
        if len(self.timewindow) == 0:
            self.timewindow.append([stuid, stutime, location])
            return
        first_time = self.timewindow[0][1]
        # if there is a stuid in timewindow same as the newone
        for record in self.timewindow:
            if stuid == record[0]:
                self.timewindow.remove(record)
                self.timewindow.append([stuid, stutime, location])
                return
        if stutime - first_time < datetime.timedelta(minutes=2):
            pass
        else:
            # remove all oldest items over 2 mins
            while (stutime - self.timewindow[0][1] >= datetime.timedelta(minutes=2)):
                self.timewindow.pop(0)
                if len(self.timewindow) == 0:
                    break
        self.__addConcurrence(stuid, location)
        self.timewindow.append([stuid, stutime, location])

    def __addConcurrence(self, stuid, location):
        stuid_index = self.stuid_index_map[stuid]
        for record in self.timewindow:
            stu1 = record[0]
            stu1_location = record[2]
            if stu1_location == location:
                stu1_index = self.stuid_index_map[stu1]
                if stu1_index < stuid_index:
                    # keep lower matrix
                    tmp = stu1_index
                    stu1_index = stuid_index
                    stuid_index = tmp
                self.cc_matrix[stuid_index, stu1_index] += 1

    def calc_matrix(self, df):
        for index, row in df.iterrows():
            try:
                # 显示进度
                if index % 1000 == 0:
                    logging.info('Social calc_matrix in process: %4.2f%%, %d in %d' % (
                    index / df.shape[0] * 100, index, df.shape[0]))
                stuid = str(row['student_id'])
                stutime = row['brush_time']
                location = row['category'] + row['location']
                self.__addOneRecord(stuid, stutime, location)
            except Exception as e:
                logging.error(e)
        # self.cc_matrix = self.cc_matrix.tocsr()
        logging.info("TimeWindow calc matrix DONE!")


def parse_location(x):
    xlocation = x['place']
    return xlocation


def calc_result(timewindow):
    intimacy_density_m = timewindow.cc_matrix

    intimacy_density_m_nonzerovalues = intimacy_density_m.tocoo().data
    mean = intimacy_density_m_nonzerovalues[intimacy_density_m_nonzerovalues >= LEAST_CO_TIMES].mean()
    std = intimacy_density_m_nonzerovalues[intimacy_density_m_nonzerovalues >= LEAST_CO_TIMES].std()
    if mean is np.nan:
        raise Exception('social calculate matrix mean is not a number!')
    if std is np.nan:
        std = 0
    # 亲密度矩阵 高斯归一化, drop 外部拟合高斯
    # intimacy_density_m = np.apply_along_axis(guassian_fit_cdf,axis=0,arr=intimacy_density_m,mean=mean,std=std)
    intimacy_density_m = intimacy_density_m.tolil()

    def max_n(row_data, row_indices, n):
        i = row_data.argsort()[-n:]
        # i = row_data.argpartition(-n)[-n:]
        top_values = row_data[i]
        top_indices = row_indices[i]  # do the sparse indices matter?
        return top_values, top_indices, i

    data_out = []
    # data_out 输出的sql中需要的字段
    # 最多取10个朋友
    # 跳过全0行
    for stuindex in range(intimacy_density_m.shape[0]):
        d, r = max_n(np.array(intimacy_density_m.data[stuindex]), np.array(intimacy_density_m.rows[stuindex]), 10)[:2]
        if stuindex % int(intimacy_density_m.shape[0]/100) == 0:
            logging.debug('Social matrix calculate friend process: %4.2f%%, %d in %d' % (stuindex / intimacy_density_m.shape[0] * 100, stuindex, intimacy_density_m.shape[0]))
        for rd_index, frd_intimacy in enumerate(d):
            # # 方式一：如果亲密度太低（小于mean + std）就不算朋友
            # if frd_intimacy < mean + std:
            #     continue
            # 方式二：每人至少2个朋友，从第三个开始，如果亲密度太低（小于mean + std）就不算朋友
            if rd_index > 2 and frd_intimacy < mean + std:
                continue
            friendindex = r[rd_index]
            stuid = timewindow.stuidlist[stuindex]
            friendid = timewindow.stuidlist[friendindex]
    return data_out


def main():
    if STU_NAME_STD_LIST.__len__() == 0 or STU_NAME_LIST.__len__() == 0 or STU_ID_STD_LIST.__len__() == 0:
        logging.error('Social Calculate Error : no student! ')
        sys.exit(1)
    if os.path.exists('tmp') == False:
        os.mkdir('tmp')
    df = pd.read_csv('../data/consum_access_feat53.csv')


    # todo : assign diff weight to types
    # consum_types = ['餐费支出', '公交支出', '用电支出', '图书馆支出', '医疗支出']

    df = df[df['student_id'].astype(str).isin(STU_ID_STD_LIST)]

    # todo: define loacation function
    df['location'] = df['place'].str.split('楼').str[0]
    # df['location'] = df.apply(parse_location, axis=1)
    if df['brush_time'].dtype == object:
        df['brush_time'] = pd.to_datetime(df['brush_time'])

    # df.to_csv('tmp/social' + end.strftime('%Y_%m') + ".csv", index=False)
    timewindow = TimeWindow(STU_ID_STD_LIST)
    # calc matrix
    timewindow.calc_matrix(df)
    # np.save('tmp/intimacy_density_matrix_%s.npy' % end.strftime('%Y-%m'), timewindow.cc_matrix)
    # load matrix
    # timewindow.cc_matrix = np.load('tmp/intimacy_density_matrix_%s.npy' % start.strftime('%Y-%m')).tolist()
    if not timewindow:
        raise Exception('Social calculate timewindow class is empty!')
    data_out = calc_result(timewindow)
    if not data_out:
        raise Exception('Social calculate result is empty!')
    df = pd.DataFrame(data_out, columns=['id', 'date', 'relation', 'type', 'friend_sn', 'sn'])
    df_all = pd.DataFrame(columns=['id', 'date', 'relation', 'type', 'friend_sn', 'sn'])
    df_all = df_all.append(df)


    mean = df_all['relation'].mean()
    std = df_all['relation'].std()
    df_all['relation'] = norm(mean, std).cdf(df_all['relation'].tolist())
    # expaned value to 0-10
    df_all['relation'] = df_all['relation'] * 10
    df_all = df_all.round(6)
    # debug
    df_all.to_csv('tmp/t_friend_out.csv', index=False)
    return 0


if __name__ == "__main__":
    # logging.basicConfig(filename='social.log',
    #                     level=logging.DEBUG,
    #                     format='%(asctime)s - %(name)s - %(threadName)s -  %(levelname)s - %(message)s')
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(threadName)s -  %(levelname)s - %(message)s')
    logging.info('job start...')
    sys.exit(main())
