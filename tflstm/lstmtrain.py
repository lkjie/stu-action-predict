#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'liwenjie'

import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import numbers
from tensorflow.python.ops import array_ops
from tensorflow.python import debug as tf_debug

# sys.path.append(os.path.dirname(os.path.realpath(__file__))+os.sep+'..')
from tflstm.dataset import build_model_columns
from tflstm.dataset import input_fn
from tflstm.logger import BaseBenchmarkLogger
from tflstm.dataset import _LABEL_NUM

FLAGS = tf.flags.FLAGS
DEBUG = FLAGS.is_debug
test_file = FLAGS.test_file
eva_file = FLAGS.eva_file
train_file = FLAGS.train_file
all_data_num = FLAGS.all_data_num
MODEL_DIR = FLAGS.model_dir
eval_hooks = []
debug_url = FLAGS.debug_url
if DEBUG:
    # train_hooks = train_hooks + [tf_debug.TensorBoardDebugHook(debug_url)]
    eval_hooks = [tf_debug.TensorBoardDebugHook(debug_url)]

batch_size = 128
num_epoch = 5
train_steps = int(2 * all_data_num * 0.7)
steps_between_evals = 1000
deep_columns = build_model_columns()
hidden_units = [1000, 500, 150, 104]
'''
batch_size, timestep_size, input_size = 128, 1 ,?
'''


def past_stop_threshold(stop_threshold, eval_metric):
    """Return a boolean representing whether a model should be stopped.
  
    Args:
      stop_threshold: float, the threshold above which a model should stop
        training.
      eval_metric: float, the current value of the relevant metric to check.
  
    Returns:
      True if training should stop, False otherwise.
  
    Raises:
      ValueError: if either stop_threshold or eval_metric is not a number
    """
    if stop_threshold is None:
        return False

    if not isinstance(stop_threshold, numbers.Number):
        raise ValueError("Threshold for checking stop conditions must be a number.")
    if not isinstance(eval_metric, numbers.Number):
        raise ValueError("Eval metric being checked against stop conditions "
                         "must be a number.")

    if eval_metric >= stop_threshold:
        tf.logging.info(
            "Stop threshold of {} was passed with metric value {}.".format(
                stop_threshold, eval_metric))
        return True

    return False


def focal_loss(prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

    # For positive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so positive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)

    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return tf.reduce_sum(per_entry_cross_ent)


def lstm_model(X, y):
    '''定义LSTM模型'''

    def lstm_cell(cell_size, keep_prob, num_proj):
        # return tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(cell_size, num_proj=min(cell_size, num_proj)), output_keep_prob=keep_prob)
        return tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(cell_size), output_keep_prob=keep_prob)

    def multi_lstm_cell(cell_sizes, keep_prob, num_proj):
        return tf.nn.rnn_cell.MultiRNNCell([lstm_cell(cell_size, keep_prob, num_proj)
                                            for cell_size in cell_sizes])

    '''以前面定义的LSTM cell为基础定义多层堆叠的LSTM，我们这里只有1层'''
    '''将已经堆叠起的LSTM单元转化成动态的可在训练过程中更新的LSTM单元'''
    cell = multi_lstm_cell(cell_sizes=hidden_units, keep_prob=0.6, num_proj=None)
    outputs, last_states = tf.nn.dynamic_rnn(
        cell=cell,
        dtype=tf.float32,
        inputs=X)

    '''根据预定义的每层神经元个数来生成隐层每个单元'''
    pred_logits = outputs[:, -1, :]

    '''通过无激活函数的全连接层计算线性回归，并将数据压缩成一维数组结构'''
    # predictions = tf.contrib.layers.fully_connected(predictions, _LABEL_NUM, None)

    '''统一预测值与真实值的形状'''
    # labels = tf.reshape(y, [-1])
    # predictions = tf.reshape(predictions, [-1])

    '''定义损失函数，这里为正常的均方误差'''
    y_onehot = tf.one_hot(y, depth=_LABEL_NUM)
    pred_logits = tf.nn.softmax(pred_logits, axis=1)
    loss = focal_loss(prediction_tensor=pred_logits, target_tensor=y_onehot)
    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=pred_logits, name='cross_entropy_loss')
    # loss = tf.reduce_mean(loss, name='focal_loss_mean')


    '''定义优化器各参数'''
    # train_op = tf.contrib.layers.optimize_loss(cost,
    #                                            tf.contrib.framework.get_global_step(),
    #                                            optimizer='Adagrad',
    #                                            learning_rate=0.2)
    train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss, global_step=tf.train.get_global_step())
    '''返回预测值、损失函数及优化器'''
    return pred_logits, loss, train_op


def model_fn(features, labels, mode, params=None):
    x = tf.feature_column.input_layer(features, params['feature_columns'])
    x = tf.reshape(x, [-1, 1, x.shape[1]])
    pred_logits, loss, train_op = lstm_model(x, labels)
    predict = tf.argmax(pred_logits, axis=1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={"result": pred_logits})
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(predictions=predict, labels=labels),
        "top@1": tf.metrics.mean(tf.nn.in_top_k(predictions=pred_logits, targets=labels, k=1), name='top1'),
        "top@3": tf.metrics.mean(tf.nn.in_top_k(predictions=pred_logits, targets=labels, k=3), name='top3'),
        "top@5": tf.metrics.mean(tf.nn.in_top_k(predictions=pred_logits, targets=labels, k=5), name='top5'),
        "top@10": tf.metrics.mean(tf.nn.in_top_k(predictions=pred_logits, targets=labels, k=10), name='top10'),
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)


class EvalStepHook(tf.train.SessionRunHook):
    def __init__(self):
        self.metrics_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="top\d{1,2}")
        self.metrics_vars_initializer = tf.variables_initializer(var_list=self.metrics_vars)

    def before_run(self, run_context):
        run_context.session.run(self.metrics_vars_initializer)


def train():
    run_config = tf.estimator.RunConfig().replace(session_config=tf.ConfigProto(device_count={"CPU": 4, 'GPU': 0},
                                                                                inter_op_parallelism_threads=0,
                                                                                intra_op_parallelism_threads=0))
    eva_input_fn = lambda: input_fn(eva_file, 1, False, batch_size)
    train_input_fn = lambda: input_fn(train_file, num_epoch, True, batch_size)
    test_input_fn = lambda: input_fn(test_file, 1, False, batch_size)
    eva_input_fn_c = lambda: input_fn(eva_file, 1, False, batch_size, is_classification=True)
    train_input_fn_c = lambda: input_fn(train_file, num_epoch, True, batch_size, is_classification=True)
    test_input_fn_c = lambda: input_fn(test_file, 1, False, batch_size, is_classification=True)
    tensors_to_log = {
        # 'average_loss': 'cross_entropy_loss_mean',
        # 'loss': 'cross_entropy_loss/cross_entropy_loss'
    }
    train_hooks = [tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)]
    # eval_hooks.append(EvalStepHook())
    # train_hooks = train_hooks + [tf_debug.LocalCLIDebugHook()]
    early_stop = False
    run_params = {
        'batch_size': batch_size,
        'train_steps': train_steps,
        'model_type': 'lstm',
        'feature_columns': deep_columns
    }
    estimator_c = tf.estimator.Estimator(model_fn=model_fn, config=run_config, params=run_params, warm_start_from=None,
                                         model_dir=MODEL_DIR)
    benchmark_logger = BaseBenchmarkLogger()
    benchmark_logger.log_run_info('wide_deep', "Census Income", run_params, test_id=None)
    metrics_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="top\d{1,2}")
    metrics_vars_initializer = tf.variables_initializer(var_list=metrics_vars)
    for n in range(train_steps // steps_between_evals):
        estimator_c.train(input_fn=train_input_fn_c, hooks=train_hooks, steps=500)
        graph = tf.get_default_graph()
        session = tf.Session(graph=graph)
        session.run(metrics_vars_initializer)
        results = estimator_c.evaluate(input_fn=eva_input_fn_c, hooks=eval_hooks)
        # Display evaluation metrics
        tf.logging.info('Results at step %d / %d', (n + 1) * steps_between_evals, train_steps)
        tf.logging.info('-' * 60)
        # for key in sorted(results):
        #     tf.logging.info('%s: %s' % (key, results[key]))
        benchmark_logger.log_evaluation_result(results)
        if early_stop and past_stop_threshold(None, results['loss']):
            break
    res = estimator_c.predict(test_input_fn_c)
    lres = [p['predictions'] for p in res]
    nres = np.array(lres)
    np.save('tflstm/nnres_c.npy', nres)
