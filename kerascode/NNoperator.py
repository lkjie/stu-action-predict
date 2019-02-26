#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'lkjie'

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
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Reshape
from keras.layers import GRU, Input, Lambda
from keras.callbacks import TensorBoard, CSVLogger, EarlyStopping
import keras
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from keras.backend.tensorflow_backend import set_session

from kerascode.configure import batch_size, epochs


def run_model(experiment, model, x_train, y_train, x_test, y_test):
    # keras_backend.set_session(tf_debug.TensorBoardDebugWrapperSession(tf.Session(), "localhost:6007"))
    tensorboard = TensorBoard(log_dir='./%s_logs' % experiment, batch_size=batch_size,
                              # embeddings_freq=5,
                              # embeddings_layer_names=emb_names,
                              # embeddings_metadata='metadata.tsv',
                              # embeddings_data=x_test
                              )
    csv_logger = CSVLogger('logs/%s_training.csv' % experiment)
    early_stopping = EarlyStopping(monitor='val_loss', patience=4)
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.2
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    set_session(tf.Session(config=config))
    model.summary()
    print('Train...')

    model.fit(x_train, y_train,
              batch_size=batch_size,
              callbacks=[tensorboard, csv_logger, early_stopping],
              epochs=epochs,
              validation_data=(x_test, y_test)
              )
    # eval_res = model.evaluate([x_test1], [y_test1], batch_size=batch_size)
    # y_p = model.predict([x_test1, x_test2])
    # print('Test evaluation:')
    # print(model.metrics_names)
    # print(eval_res)
    # print(y_p)
    model.save('models/%s_model' % experiment)
