#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'liwenjie'

import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import numbers
import json
from tensorflow.python.ops import array_ops
from tensorflow.python import debug as tf_debug
sys.path.append('/home/liwenjie/liwenjie/projects/lwjpaper')

'''
batch_size can ONLY set to 1
'''
all_data_num = 41633
_LABEL_NUM = 104
CHECKPOINT_PATH = '/home/liwenjie/liwenjie/tftmp/lstmlow0/'

padding_size = 900
batch_size = 1
num_epoch = 2
train_steps = int(2 * all_data_num * 0.7)
steps_between_evals = 1000
hidden_units = [1000, 500, 150, 104]

unique_file = '/home/liwenjie/liwenjie/projects/lwjpaper/data/unique.json'
categorical_features = ['student_id', 'card_id', 'timeslot', 'trans_type', 'category']
continus_features = ['amount', 'remained_amount']
unique_value = json.load(open(unique_file))
_HASH_BUCKET_SIZE = 1000
validation_metrics_var_scope = "validation_metrics"


def variable_summaries(var, name=''):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('%s_mean' % name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('%s_stddev' % name, stddev)
    tf.summary.scalar('%s_max' % name, tf.reduce_max(var))
    tf.summary.scalar('%s_min' % name, tf.reduce_min(var))
    tf.summary.histogram('%s_histogram' % name, var)


def build_model_columns():
    """Builds a set of wide and deep feature columns."""
    # Continuous variable columns
    cate_columns = []
    cate_embedding_columns = []
    cate_no_embedding_columns = []
    continus_columns = []
    embedding_features = []
    for col in categorical_features:
        coluniquelen = len(unique_value[col])
        if coluniquelen > _HASH_BUCKET_SIZE:
            print("col name :{}\t\tcount:{}".format(col, coluniquelen))
            embedding_features.append(col)
    for feat in categorical_features:
        f = tf.feature_column.categorical_column_with_vocabulary_list(feat, unique_value[feat])
        if feat in embedding_features:
            f1 = tf.feature_column.embedding_column(f, dimension=100)
            cate_embedding_columns.append(f1)
        else:
            f1 = tf.feature_column.indicator_column(f)
            cate_no_embedding_columns.append(f1)
        cate_columns.append(f)
    for feat in continus_features:
        f = tf.feature_column.numeric_column(feat)
        continus_columns.append(f)
    deep_columns = cate_embedding_columns + cate_no_embedding_columns + continus_columns
    return deep_columns


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


def lstm_model(X, y, X_length):
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
        inputs=X,
        sequence_length=X_length)

    '''根据预定义的每层神经元个数来生成隐层每个单元'''
    pred_logits = outputs[:, :X_length[0], :]

    '''通过无激活函数的全连接层计算线性回归，并将数据压缩成一维数组结构'''
    # predictions = tf.contrib.layers.fully_connected(predictions, _LABEL_NUM, None)

    '''统一预测值与真实值的形状'''
    # labels = tf.reshape(y, [-1])
    # predictions = tf.reshape(predictions, [-1])

    '''定义损失函数，这里为正常的均方误差'''
    y = tf.reshape(y, [batch_size*X_length[0]])
    pred_logits = tf.reshape(pred_logits, [batch_size*X_length[0], _LABEL_NUM])
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


def model_fn(features, labels, feature_length, feature_columns=None):
    '''
    batchsize must be 1
    :param features: 
    :param labels: 
    :param feature_length: 
    :param feature_columns: 
    :return: 
    '''
    # features = tf.reshape(features, [-1, batch_size*padding_size])
    # feature_timelength = features['student_id'].shape[0].value
    # for col in features.keys():
    #     features[col] = tf.reshape(features[col], [-1, 1])
    deep_columns = build_model_columns()
    x = tf.feature_column.input_layer(features, deep_columns)
    variable_summaries(x, name='inputLayer_x')
    x = tf.reshape(x, [1, feature_length, 257])
    x = tf.pad(x, [[0, 0], [0, padding_size-tf.shape(x)[1]], [0, 0]])
    x.set_shape([1, padding_size, 257])
    # x = tf.set(x, )
    feature_length_list = tf.reshape(feature_length, shape=(1,))
    pred_logits, loss, train_op = lstm_model(x, labels, feature_length_list)
    pred_logits = tf.reshape(pred_logits, [feature_length, _LABEL_NUM])
    predict = tf.argmax(pred_logits, axis=1)
    labels = tf.reshape(labels, [-1])
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(predictions=predict, labels=labels, name=validation_metrics_var_scope),
        "top@1": tf.metrics.mean(tf.nn.in_top_k(predictions=pred_logits, targets=labels, k=1), name=validation_metrics_var_scope),
        "top@3": tf.metrics.mean(tf.nn.in_top_k(predictions=pred_logits, targets=labels, k=3), name=validation_metrics_var_scope),
        "top@5": tf.metrics.mean(tf.nn.in_top_k(predictions=pred_logits, targets=labels, k=5), name=validation_metrics_var_scope),
        "top@10": tf.metrics.mean(tf.nn.in_top_k(predictions=pred_logits, targets=labels, k=10), name=validation_metrics_var_scope),
    }
    tf.summary.scalar('loss', loss)
    for key in eval_metric_ops.keys():
        eval_value, eval_value_op = eval_metric_ops[key]
        tf.summary.scalar(key, eval_value)
    return pred_logits, loss, train_op, eval_metric_ops


def model_fn2(features, labels, feature_length, feature_columns=None):
    '''
    batchsize must be 1
    :param features: 
    :param labels: 
    :param feature_length: 
    :param feature_columns: 
    :return: 
    '''
    deep_columns = build_model_columns()
    # features = tf.reshape(features, [-1, batch_size*padding_size])
    # feature_timelength = features['student_id'].shape[0].value
    # for col in features.keys():
    #     features[col] = tf.reshape(features[col], [-1, 1])
    x = tf.feature_column.input_layer(features, deep_columns)
    return x


def pad_to_dense(M, fill_value, dtype=np.float32):
    """Appends the minimal required amount of zeroes at the end of each 
     array in the jagged array `M`, such that `M` looses its jagedness."""

    maxlen = max(len(r) for r in M)

    Z = np.full(shape=(len(M), maxlen), fill_value=fill_value, dtype=dtype)
    for enu, row in enumerate(M):
        Z[enu, :len(row)] += row
    return Z


def get_feed(sid, df_consum, place_holders):
    features_placeholder, labels_placeholder, feature_length_placeholder = place_holders
    dic_feat = {}
    stuconsum = df_consum[df_consum['student_id'] == sid]
    for i, col in enumerate(df_consum.columns):
        if col == 'placei':
            dic_feat[labels_placeholder] = stuconsum[col].values
        else:
            dic_feat[features_placeholder[i]] = stuconsum[col].values
    dic_feat[feature_length_placeholder] = stuconsum['student_id'].shape[0]
    return dic_feat


def main():
    initializer = tf.random_uniform_initializer(-.05, .05)
    consum = pd.read_csv('../data/consum_cleaned.csv')
    consum['card_id'] = consum['card_id'].apply(str)
    stutimes = consum.groupby(['student_id']).count()['card_id']
    stuid = stutimes[(stutimes > 100) & (stutimes < padding_size)].index
    stuid = pd.Series(stuid)
    stuid_train = stuid.sample(frac=0.7)
    stuid_test = stuid[~stuid.isin(stuid_train)]
    features_placeholder_dict = {}
    features_placeholder = []
    labels_placeholder = tf.placeholder(consum['placei'].dtype, shape=(None), name="placei")
    for col in consum.columns.drop('placei'):
        feature_placeholder = tf.placeholder(consum[col].dtype, shape=(None), name=col)
        features_placeholder.append(feature_placeholder)
        features_placeholder_dict[col] = feature_placeholder
    feature_length_placeholder = tf.placeholder(tf.int32, name='feature_length')
    pred_logits, loss, train_op, eval_metric_ops = model_fn(features_placeholder_dict, labels_placeholder, feature_length_placeholder)
    place_holders = features_placeholder, labels_placeholder, feature_length_placeholder
    merged = tf.summary.merge_all()
    # x = model_fn(features_placeholder_dict, labels_placeholder, feature_length_placeholder)
    saver = tf.train.Saver()
    step = 0

    with tf.Session() as sess:
        # validation metric init op
        validation_metrics_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=validation_metrics_var_scope)
        validation_metrics_init_op = tf.variables_initializer(var_list=validation_metrics_vars, name='validation_metrics_init')
        tf.initialize_all_tables().run()
        sess.run(validation_metrics_init_op)
        train_writer = tf.summary.FileWriter(CHECKPOINT_PATH + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(CHECKPOINT_PATH + '/test')
        tf.global_variables_initializer().run()
        # sess = tf_debug.TensorBoardDebugWrapperSession(sess, "172.16.170.116:6066")
        for i in range(num_epoch):
            print("in epoch: %d" % (i+1))
            for sid in stuid_train:
                dic_feat = get_feed(sid, consum, place_holders)
                summary, cost, _ = sess.run([merged, loss, train_op], feed_dict=dic_feat)
                train_writer.add_summary(summary, step)
                if step % 10 == 0:
                    print("after %d steps, per token cost is %.3f" % (step, cost))
                if step % 50 == 0:
                    saver.save(sess, CHECKPOINT_PATH, global_step=step)
                    sid_test = stuid_test.sample(n=1).iloc[0]
                    dic_feat_test = get_feed(sid_test, consum, place_holders)
                    summary_test, cost = sess.run([merged, loss], feed_dict=dic_feat_test)
                    test_writer.add_summary(summary_test, step)
                step += 1


if __name__ == '__main__':
    main()