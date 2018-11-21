#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'liwenjie'

import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf

os.chdir('/home/liwenjie/liwenjie/projects/lwjpaper')
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('train_id', None, 'train function id.')


def train_0():
    '''
    训练标签为当前时间
    :return: 
    '''
    tf.flags.DEFINE_string('test_file', 'data/tf_test.csv', 'Directory for testing data.')
    tf.flags.DEFINE_string('eva_file', 'data/tf_val.csv', 'Directory for evaluating data.')
    tf.flags.DEFINE_string('train_file', 'data/tf_train.csv', 'Directory for training data.')
    tf.flags.DEFINE_string('whole_file', 'data/consum_cleaned.csv', 'Directory for whole data')
    tf.flags.DEFINE_integer('all_data_num', 17827913, 'whole data entries.')
    tf.flags.DEFINE_string('model_dir', '/home/liwenjie/liwenjie/tftmp/stulstm_focalloss',
                           'Directory to keep the model.')
    tf.flags.DEFINE_string('debug_url', "localhost:6068", 'Address and port for debug.')
    tf.flags.DEFINE_bool('is_debug', False, 'Whether debug the model.')
    tf.flags.DEFINE_string('label_name', 'placei', 'Name of label in csv.')
    tf.flags.DEFINE_integer('label_cate_num', 104, 'Number of label category')
    from tflstm.lstmtrain import train
    model_dir = FLAGS.model_dir
    # os.system('rm -rf %s_bak' % model_dir)
    # os.system('mv %s %s' % (model_dir, model_dir))
    train()


def train_label_next():
    '''
    训练标签为下次时间
    :return: 
    '''
    tf.flags.DEFINE_string('test_file', 'data/tf_test_label_next.csv', 'Directory for testing data.')
    tf.flags.DEFINE_string('eva_file', 'data/tf_val_label_next.csv', 'Directory for evaluating data.')
    tf.flags.DEFINE_string('train_file', 'data/tf_train_label_next.csv', 'Directory for training data.')
    tf.flags.DEFINE_string('whole_file', 'data/consum_cleaned_label_next.csv', 'Directory for whole data')
    tf.flags.DEFINE_integer('all_data_num', 16878599, 'whole data entries.')
    tf.flags.DEFINE_string('model_dir', '/home/liwenjie/liwenjie/tftmp/stulstm_focalloss_label_next',
                           'Directory to keep the model.')
    tf.flags.DEFINE_string('debug_url', "localhost:6068", 'Address and port for debug.')
    tf.flags.DEFINE_bool('is_debug', False, 'Whether debug the model.')
    tf.flags.DEFINE_string('label_name', 'place_next', 'Name of label in csv.')
    tf.flags.DEFINE_integer('label_cate_num', 104, 'Number of label category')
    from tflstm.lstmtrain import train
    model_dir = FLAGS.model_dir
    # os.system('rm -rf %s_bak' % model_dir)
    # os.system('mv %s %s_bak' % (model_dir, model_dir))
    train()


def train_2():
    '''
    训练标签为当前时间
    loss: cross entropy
    :return: 
    '''
    tf.flags.DEFINE_string('test_file', 'data/tf_test.csv', 'Directory for testing data.')
    tf.flags.DEFINE_string('eva_file', 'data/tf_val.csv', 'Directory for evaluating data.')
    tf.flags.DEFINE_string('train_file', 'data/tf_train.csv', 'Directory for training data.')
    tf.flags.DEFINE_string('whole_file', 'data/consum_cleaned.csv', 'Directory for whole data')
    tf.flags.DEFINE_integer('all_data_num', 17827913, 'whole data entries.')
    tf.flags.DEFINE_string('model_dir', '/home/liwenjie/liwenjie/tftmp/stulstm_focalloss',
                           'Directory to keep the model.')
    tf.flags.DEFINE_string('debug_url', "localhost:6068", 'Address and port for debug.')
    tf.flags.DEFINE_bool('is_debug', False, 'Whether debug the model.')
    tf.flags.DEFINE_string('label_name', 'placei', 'Name of label in csv.')
    tf.flags.DEFINE_integer('label_cate_num', 104, 'Number of label category')
    from tflstm.lstmtrain import train
    model_dir = FLAGS.model_dir
    # os.system('rm -rf %s_bak' % model_dir)
    # os.system('mv %s %s' % (model_dir, model_dir))
    train()


def train_3():
    '''
    训练标签为下次时间
    loss: cross entropy
    :return: 
    '''
    tf.flags.DEFINE_string('test_file', 'data/tf_test_label_next.csv', 'Directory for testing data.')
    tf.flags.DEFINE_string('eva_file', 'data/tf_val_label_next.csv', 'Directory for evaluating data.')
    tf.flags.DEFINE_string('train_file', 'data/tf_train_label_next.csv', 'Directory for training data.')
    tf.flags.DEFINE_string('whole_file', 'data/consum_cleaned_label_next.csv', 'Directory for whole data')
    tf.flags.DEFINE_integer('all_data_num', 16878599, 'whole data entries.')
    tf.flags.DEFINE_string('model_dir', '/home/liwenjie/liwenjie/tftmp/stulstm_focalloss_label_next',
                           'Directory to keep the model.')
    tf.flags.DEFINE_string('debug_url', "localhost:6068", 'Address and port for debug.')
    tf.flags.DEFINE_bool('is_debug', False, 'Whether debug the model.')
    tf.flags.DEFINE_string('label_name', 'place_next', 'Name of label in csv.')
    tf.flags.DEFINE_integer('label_cate_num', 104, 'Number of label category')
    from tflstm.lstmtrain import train
    model_dir = FLAGS.model_dir
    # os.system('rm -rf %s_bak' % model_dir)
    # os.system('mv %s %s_bak' % (model_dir, model_dir))
    train()


def main(_):
    if FLAGS.train_id == 0:
        train_0()
    elif FLAGS.train_id == 1:
        train_label_next()
    else:
        raise ValueError('Only support 0 or 1.')


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
