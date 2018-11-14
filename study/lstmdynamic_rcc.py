#coding=utf-8
import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat
tf.estimator.Estimator
# 创建输入数据 (batchsize, )
X = np.random.randn(2, 10, 8)

# 第二个example长度为6
X[1,6:] = 0
X_lengths = [10, 6]

cell = tf.nn.rnn_cell.LSTMCell(num_units=64, state_is_tuple=True)

outputs, last_states = tf.nn.dynamic_rnn(
    cell=cell,
    dtype=tf.float64,
    sequence_length=X_lengths,
    inputs=X)
n_iter = 10
result = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(n_iter):
        result_ = sess.run({"outputs": outputs, "last_states": last_states}, feed_dict=None)
        result.append(result_)

print(result[0])
assert result[0]["outputs"].shape == (2, 10, 64)

# 第二个example中的outputs超过6步(7-10步)的值应该为0
assert (result[0]["outputs"][1,7,:] == np.zeros(cell.output_size)).all()
