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
import re
from sqlalchemy import create_engine
plt.style.use('fivethirtyeight')
import warnings
import matplotlib.pyplot as plt
sns.set(font='SimHei')

f, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
f, ax = plt.subplots(figsize=(30, 30))