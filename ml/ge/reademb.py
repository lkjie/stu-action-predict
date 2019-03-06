import io
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_array
import tensorflow as tf
import numpy as np
import os

# from kerascode.configure import *

l = []
bd_vec_file = '/home/liwenjie/liwenjie/projects/lwjpaper/ml/ge/vectors_new_uniform/net_bd_vec.txt'
cat_vec_file = '/home/liwenjie/liwenjie/projects/lwjpaper/ml/ge/vectors_new_uniform/net_cat_vec.txt'
time_vec_file = '/home/liwenjie/liwenjie/projects/lwjpaper/ml/ge/vectors_new_uniform/net_time_vec.txt'
user_vec_file = '/home/liwenjie/liwenjie/projects/lwjpaper/ml/ge/vectors_new_uniform/net_user_vec.txt'

def read_file(f):
    file = open(f)
    count, vec_len = map(int, file.readline().strip().split())
    stu_ids = []
    vecs = []
    for i in range(count):
        stu_id = file.readline().strip()
        stu_ids.append(stu_id)
        vecs.append(file.readline().strip().split())
    df = pd.DataFrame(vecs, index=stu_ids, dtype=np.float32)
    return df


bd_vec = read_file(bd_vec_file)
cat_vec = read_file(cat_vec_file)
time_vec = read_file(time_vec_file)
user_vec = read_file(user_vec_file)
# stu_map = pd.read_csv('/home/liwenjie/liwenjie/projects/lwjpaper/data/consum_access_feat6m_stuids.csv')
consum = pd.read_csv('/home/liwenjie/liwenjie/projects/lwjpaper/ml/ge/test.csv')
# consum = consum.merge(stu_map, how='left', on='student_id_int')
consum = consum[consum.student_id != '2013060108021']
print('Loading data...')

features = ['student_id', 'timeslot_week']
labels = ['placei']

x_test = consum[features]
y_test = consum[labels]
print(len(y_test), 'test sequences')

BU = np.matmul(user_vec.values, bd_vec.values.T)
BT = np.matmul(time_vec.values, bd_vec.values.T)

results = {}
for user in user_vec.index:
    stu_bds = {}
    for ti in time_vec.index:
        ui = user_vec.index.get_loc(user)
        ti = int(ti)
        stu_bds[ti] = BU[ui]*BT[ti]
    results[user] = stu_bds

def predict(X):
    res = []
    X = check_array(X, dtype=[str, int, float])
    for i in range(X.shape[0]):
        user, time = X[i]
        time = int(time)
        res.append(results[user][time])
    return np.array(res)

y_pred = predict(x_test)

config = tf.ConfigProto(device_count={"CPU": 1, 'GPU': 0})
tfsess = tf.Session(config=config)

def mertics_acck(y_true, y_pred):
    def topk_tf(ytrue, ypred, k=1):
        assert len(ytrue.shape) == 1 and len(ypred.shape) == 2
        predictions = tf.Variable(ypred, dtype=tf.float32)
        targets = tf.Variable(ytrue)
        topk = tf.nn.in_top_k(
            predictions,
            targets,
            k,
            name=None
        )
        tfsess.run(predictions.initializer)
        tfsess.run(targets.initializer)
        res = tfsess.run(topk)
        return res.sum() / res.shape[0]

    init_op = tf.global_variables_initializer()
    tfsess.run(init_op)
    top1 = topk_tf(y_true, y_pred, k=1)
    top3 = topk_tf(y_true, y_pred, k=3)
    top5 = topk_tf(y_true, y_pred, k=5)
    top10 = topk_tf(y_true, y_pred, k=10)
    print('top1, top3, top5, top10')
    print(top1, top3, top5, top10)

mertics_acck(y_test.values.flatten(), y_pred)

