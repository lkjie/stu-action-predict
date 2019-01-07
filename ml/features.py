#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'liwenjie'


import pandas as pd
import pymysql, logging, os
import sys, datetime
from sqlalchemy import create_engine
import numpy as np
from dateutil.relativedelta import relativedelta
from scipy.stats import norm
import matplotlib.pyplot as plt
import queue, json, time, dateutil, math
from collections import namedtuple
import json


DB_OUT_HOST = '172.16.170.81'
DB_OUT_PORT = 3307
DB_OUT_USER = 'liwenjie'
DB_OUT_PWD = 'liwenjie333'
DB_OUT_NAME = 'data_center'


engine = create_engine('mysql+pymysql://%(user)s:%(pwd)s@%(host)s:%(port)s/%(dbname)s?charset=utf8' % {'user': DB_OUT_USER, 'pwd': DB_OUT_PWD, 'host': DB_OUT_HOST, 'dbname': DB_OUT_NAME, 'port':DB_OUT_PORT},
                                       encoding='utf-8')
conn = engine.connect()
consum = pd.read_csv('../data/consumption.txt')
aimstudent = consum['student_id'].drop_duplicates()

# 基本信息
df_basicinfo = pd.read_sql_table('t_basicinfo', conn)


# 选课信息
df_course_selection = pd.read_sql_table('t_course_selection', conn)

# 图书借阅
df_libraryborrow = pd.read_sql_table('t_libraryborrow', conn)

