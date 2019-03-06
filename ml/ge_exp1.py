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


'''
基于异步图构建来进行推荐
数据来源：
/home/liwenjie/liwenjie/projects/lwjpaper/data/poi_activity.csv
/home/liwenjie/liwenjie/projects/lwjpaper/data/poi_time.csv
/home/liwenjie/liwenjie/projects/lwjpaper/data/poi_user.csv
'''
consum = pd.read_csv('../data/consum_access_feat6m.csv')
features = ['student_id_int', 'timeslot_week']
labels = ['placei']

