import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

# TensorFlow的高层封装TFLearn
learn = tf.contrib.learn

# 神经网络参数
HIDDEN_SIZE = 30  # LSTM隐藏节点个数
NUM_LAYERS = 2  # LSTM层数
TIMESTEPS = 10  # 循环神经网络截断长度
BATCH_SIZE = 32  # batch大小

# 数据参数
TRAINING_STEPS = 3000  # 训练轮数
TRAINING_EXAMPLES = 10000  # 训练数据个数
TESTING_EXAMPLES = 1000  # 测试数据个数
SAMPLE_GAP = 0.01  # 采样间隔


def generate_data(seq):
    # 序列的第i项和后面的TIMESTEPS-1项合在一起作为输入，第i+TIMESTEPS项作为输出
    # X = []
    # y = []
    # for i in range(len(seq) - TIMESTEPS - 1):
    #     X.append([seq[i:i + TIMESTEPS]])
    #     y.append([seq[i + TIMESTEPS]])
    # train_X, train_y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    train_X = np.random.rand(BATCH_SIZE,TIMESTEPS,6,8)
    train_y = np.random.rand(BATCH_SIZE,TIMESTEPS)
    return train_X, train_y


def lstm_model(X, y, mode):
    # 使用多层的LSTM结构
    def lstm_cell(cell_size, keep_prob, num_proj):
        return tf.nn.rnn_cell.DropoutWrapper(
            tf.nn.rnn_cell.LSTMCell(cell_size, num_proj=min(cell_size, num_proj)),
            output_keep_prob=keep_prob)

    def multi_lstm_cell(cell_sizes, keep_prob, num_proj):
        return tf.nn.rnn_cell.MultiRNNCell([lstm_cell(cell_size, keep_prob, num_proj)
                                            for cell_size in cell_sizes])

    '''以前面定义的LSTM cell为基础定义多层堆叠的LSTM，我们这里只有1层'''
    '''将已经堆叠起的LSTM单元转化成动态的可在训练过程中更新的LSTM单元'''
    cell = multi_lstm_cell(cell_sizes=[64, 64, 64], keep_prob=0.6, num_proj=64)
    # 使用TensorFlow接口将多层的LSTM结构连接成RNN网络并计算其前向传播结果
    outputs, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    output = outputs[:, -1, :]
    # 对LSTM网络的输出再做加一层全链接层并计算损失，注意这里默认的损失为平均
    # 平方差损失函数
    predictions = tf.contrib.layers.fully_connected(output, 1, activation_fn=None)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return predictions, None, None
    # 计算损失函数
    loss = tf.losses.mean_squared_error(labels=y, predictions=predictions)
    if mode == tf.estimator.ModeKeys.EVAL:
        return predictions, loss, None
    # 创建模型优化器并得到优化步骤
    train_op = tf.contrib.layers.optimize_loss(loss, tf.train.get_global_step(), optimizer="Adagrad", learning_rate=0.1)
    # 只在训练时计算损失函数和优化步骤，测试时直接返回预测结果
    return predictions, loss, train_op


def model_fn(features, labels, mode, params=None):
    predict, loss, train_op = lstm_model(features, labels, mode=mode)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={"result": predict})
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(predict, labels)
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)


# 用sin生成训练和测试数据集
test_start = TRAINING_EXAMPLES * SAMPLE_GAP
test_end = (TRAINING_EXAMPLES + TESTING_EXAMPLES) * SAMPLE_GAP
train_X, train_y = generate_data(np.sin(np.linspace(0, test_start, TRAINING_EXAMPLES, dtype=np.float32)))
test_X, test_y = generate_data(np.sin(np.linspace(test_start, test_end, TESTING_EXAMPLES, dtype=np.float32)))

model_params = {'learning_rate': 0.01}
estimator = tf.estimator.Estimator(model_fn=model_fn, params=model_params)
train_input_fn = tf.estimator.inputs.numpy_input_fn(x=train_X, y=train_y, num_epochs=None, batch_size=128, shuffle=True)
test_input_fn = tf.estimator.inputs.numpy_input_fn(x=test_X, y=test_y, num_epochs=1, batch_size=128, shuffle=False)
predict_input_fn = tf.estimator.inputs.numpy_input_fn(x=test_X, num_epochs=1, shuffle=False)
estimator.train(input_fn=train_input_fn, steps=3000)
test_results = estimator.evaluate(input_fn=test_input_fn)
predictions = estimator.predict(input_fn=predict_input_fn)
predicted = [pred['result'] for pred in predictions]

# 计算rmse作为评价指标
rmse = np.sqrt(((predicted - test_y) ** 2).mean(axis=0))
print('Mean Square Error is: %f' % (rmse[0]))

# 对预测曲线绘图，并存储到sin.jpg
fig = plt.figure()
plot_predicted = plt.plot(predicted, label='predicted')
plot_test, = plt.plot(test_y, label='real_sin')
plt.legend([plot_predicted, plot_test], ['predicted', 'real_sin'])
plt.show()
