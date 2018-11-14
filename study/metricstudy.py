#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'liwenjie'

import tensorflow as tf
import numpy as np

y_true = np.array([[2], [1], [0], [3], [0]]).astype(np.int64)
y_true = tf.identity(y_true)

y_pred = np.array([[0.1, 0.2, 0.6, 0.1],
                   [0.8, 0.05, 0.1, 0.05],
                   [0.3, 0.4, 0.1, 0.2],
                   [0.6, 0.25, 0.1, 0.05],
                   [0.1, 0.2, 0.6, 0.1]
                   ]).astype(np.float32)
y_pred = tf.identity(y_pred)

_, m_ap = tf.metrics.sparse_average_precision_at_k(y_true, y_pred, 3)

sess = tf.Session()
sess.run(tf.local_variables_initializer())

stream_vars = [i for i in tf.local_variables()]

tf_map = sess.run(m_ap)
print("TF_MAP",tf_map)

print("STREAM_VARS",(sess.run(stream_vars)))

tmp_rank = tf.nn.top_k(y_pred,3)

print("TMP_RANK",sess.run(tmp_rank))