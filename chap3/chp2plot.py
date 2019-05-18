import os
import sys
from dateutil.parser import parse
import pandas as pd
import numpy as np
import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sqlalchemy import create_engine
plt.style.use('fivethirtyeight')
import warnings
import matplotlib.pyplot as plt
sns.set(font='SimHei')
import logging
logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(threadName)s -  %(levelname)s - %(message)s')
logging.info('start...')

arr = np.arange(-3,3,0.02)
relu = np.where(arr < 0, 0, arr)
leakyrelu = np.where(arr < 0, 0.1*arr, arr)
sigmoid = 1/(1+np.power(np.e, -arr))
softplus = np.log(1+np.power(np.e, arr))
tanh = np.tanh(arr)
paper_rc = {'lines.linewidth': 3}
sns.set_context(rc = paper_rc)
sns.set(font='SimHei')
f, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.lineplot(x=arr,y=relu,ax=ax, label="ReLU", linewidth=3)
sns.lineplot(x=arr,y=sigmoid,ax=ax, label="sigmoid", linewidth=3)
sns.lineplot(x=arr,y=softplus,ax=ax, label="softplus", linewidth=3)
sns.lineplot(x=arr,y=tanh,ax=ax, label="tanh", linewidth=3)
sns.lineplot(x=arr,y=leakyrelu,ax=ax, label="Leaky ReLU", linewidth=3)

ax.set(xlabel='x', ylabel='y')
ax.set_title("常用的激活函数")
