#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'lkjie'

import pandas as pd
import pymysql, logging, os
import sys, datetime
import numpy as np
import matplotlib.pyplot as plt
import queue, json, time, dateutil, math
import json
import keras
import tensorflow as tf
import warnings

from collections import namedtuple
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Reshape
from keras.layers import GRU, Input, Lambda
from keras.callbacks import TensorBoard, CSVLogger, EarlyStopping
from tensorflow.python import debug as tf_debug
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import Callback
from sqlalchemy import create_engine
from dateutil.relativedelta import relativedelta
from scipy.stats import norm

'''
工具文件，不允许引用其他内部包，防止交叉引用
'''

class MyEarlyStopping(Callback):
    """
    Extended from Keras.callbacks.EarlyStopping for multi-monitors
    Stop training when a monitored quantity has stopped improving.

    # Arguments
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        mode: one of {auto, min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        baseline: Baseline value for the monitored quantity to reach.
            Training will stop if the model doesn't show improvement
            over the baseline.
        restore_best_weights: whether to restore model weights from
            the epoch with the best value of the monitored quantity.
            If False, the model weights obtained at the last step of
            training are used.
    """

    def __init__(self,
                 monitors=None,
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baselines=None,
                 restore_best_weights=False):
        super(MyEarlyStopping, self).__init__()

        if monitors is None:
            monitors = ['val_loss']
        self.monitors = monitors
        self.baselines = baselines
        self.patience = patience
        self.verbose = verbose
        self.waits = [0 for _ in monitors]
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % mode,
                          RuntimeWarning)
            mode = 'auto'

        self.monitor_ops = []
        self.min_deltas = []
        for monitor in self.monitors:
            if mode == 'min':
                monitor_op = np.less
            elif mode == 'max':
                monitor_op = np.greater
            else:
                if 'acc' in monitor:
                    monitor_op = np.greater
                else:
                    monitor_op = np.less

            if monitor_op == np.greater:
                md = min_delta * 1
            else:
                md = min_delta * -1
            self.monitor_ops.append(monitor_op)
            self.min_deltas.append(md)

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.waits = [0 for _ in self.monitors]
        self.stopped_epoch = 0
        if self.baselines is not None:
            self.bests = self.baselines
        else:
            self.bests = [np.Inf if m == np.less else -np.Inf for m in self.monitor_ops]

    def on_epoch_end(self, epoch, logs=None):
        current_list = self.get_monitor_value(logs)
        if current_list is None:
            return

        new_bests = []
        new_waits = []
        for monitor_op, min_delta, best, wait, current in zip(self.monitor_ops, self.min_deltas, self.bests, self.waits, current_list):
            if monitor_op(current - min_delta, best):
                best = current
                wait = 0
                if self.restore_best_weights:
                    self.best_weights = self.model.get_weights()
            else:
                wait += 1
            new_bests.append(best)
            new_waits.append(wait)
        self.waits = new_waits
        self.bests = new_bests
        if np.greater(np.array(self.waits), self.patience).all():
            self.stopped_epoch = epoch
            self.model.stop_training = True
            if self.restore_best_weights:
                if self.verbose > 0:
                    print('Restoring model weights from the end of '
                          'the best epoch')
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

    def get_monitor_value(self, logs):
        monitors_value = [logs.get(m) for m in self.monitors]
        if monitors_value is None:
            warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitors, ','.join(list(logs.keys()))), RuntimeWarning
            )
        return monitors_value


def run_model(experiment, model, x_train, y_train, x_test, y_test, early_stop='standard', embedding_log=False,
              embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, batch_size=32, epochs=40):
    print('Task name: %s' % experiment)
    # keras_backend.set_session(tf_debug.TensorBoardDebugWrapperSession(tf.Session(), "localhost:6007"))
    if embedding_log:
        tensorboard = TensorBoard(log_dir='./%s_logs' % experiment, batch_size=batch_size,
                                  embeddings_freq=5,
                                  embeddings_layer_names=embeddings_layer_names,
                                  embeddings_metadata=embeddings_metadata,
                                  embeddings_data=embeddings_data
                                  )
    else:
        tensorboard = TensorBoard(log_dir='./%s_logs' % experiment, batch_size=batch_size)
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.2
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    set_session(tf.Session(config=config))
    model.summary()
    print('Train...')
    csv_logger = CSVLogger('logs/%s_training.csv' % experiment)
    if early_stop == 'standard':
        early_stopping = EarlyStopping(monitor='val_loss', patience=4)
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  callbacks=[tensorboard, csv_logger, early_stopping],
                  epochs=epochs,
                  validation_data=(x_test, y_test)
                  )
    elif early_stop == 'multi':
        early_stopping = MyEarlyStopping(monitors=['val_loss', 'val_out_place_loss', 'val_out_amount_loss'], patience=4,
                                         verbose=1)
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  callbacks=[tensorboard, csv_logger, early_stopping],
                  epochs=epochs,
                  validation_data=(x_test, y_test)
                  )
    else:
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  callbacks=[tensorboard, csv_logger],
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
