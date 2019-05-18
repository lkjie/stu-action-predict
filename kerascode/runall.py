#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'lkjie'

import os, sys
import subprocess
import time

root_dir = '/home/liwenjie/liwenjie/projects/lwjpaper/kerascode'
files = os.listdir(root_dir)

exclude_files = ['runall.py', 'test.py', 'NNUtils.py', 'imdb_lstm.py', 'configure.py']
run_files = [
    # 'exp3_GRU1.py',
    # 'exp3_GRU.py',
    # 'exp3.py',
    'exp5.py',
    # 'exp6.py',
    # 'exp7.py',
    'exp8.py',
    # 'exp9_GRU1_128_focalloss.py',
    # 'exp9_GRU1_512_focalloss.py',
    # 'exp9_GRU1_emb12.py',
    # 'exp9_GRU1_emb32.py',
    # 'exp9_GRU1_emb64.py',
    # 'exp9_GRU1_focalloss_noalpha.py',
    # 'exp9_GRU1_focalloss_onlypos.py',
    # 'exp9_GRU1_focalloss.py',
    # 'exp9_GRU1_focalloss_sgd.py',
    # 'exp9_GRU1.py',
    # 'exp9_GRU2.py',
    # 'exp9_lstm1.py',
    # 'exp9_lstm3 .py',
    # 'exp9.py',
    # 'exp10_GRU1.py',
    # 'exp11_GRU1.py',
    # 'exp11_GRU1_relu.py',
    'exp12_GRU1.py',
    'exp13_GRU1.py',
]

files = list(filter(lambda f: os.path.isfile(f) and f not in exclude_files and f.endswith('.py'), files))

bsize = 4

# for i in range(0, len(files), bsize):
#     processes = []
#     time_start = time.time()
#     if i + bsize > len(files):
#         batch = files[i:]
#     else:
#         batch = files[i:i + bsize]
#     for f in batch:
#         cmd = 'python %s' % f
#         out = open('%s/bashlogs/%s.log' % (root_dir, f.rstrip('.py')), 'w')
#         print('run %s...' % f)
#         p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=out, stderr=out, shell=True)
#         processes.append(p)
#     [p.wait() for p in processes]
#     time_end = time.time()
#     print('this tasks is done, in %d seconds : ' % int(time_end - time_start), batch)

# 优化版本
if not os.path.exists('bashlogs'):
    os.mkdir('bashlogs')
processes = {}
error_works = []
for f in files:
    if f not in run_files:
        continue
    cmd = 'python %s' % f
    workname = f.rstrip('.py')
    out = open('%s/bashlogs/%s.log' % (root_dir, workname), 'w')
    print('run %s...' % f)
    time_start = time.time()
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=out, stderr=out, shell=True)
    processes[workname] = (time_start, p)
    while len(processes) >= bsize:
        end_work = []
        for workname, pv in processes.items():
            time_start, p = pv
            if p.poll() is not None:
                time_end = time.time()
                print('the %s is done, takes %d seconds.' % (workname, int(time_end - time_start)))
                end_work.append(workname)
                if p.poll() != 0:
                    error_works.append(workname)
        for w in end_work:
            processes.pop(w)
        time.sleep(10)

while len(processes) > 0:
    end_work = []
    for workname, pv in processes.items():
        time_start, p = pv
        if p.poll() is not None:
            time_end = time.time()
            print('the %s is done, takes %d seconds.' % (workname, int(time_end - time_start)))
            end_work.append(workname)
            if p.poll() != 0:
                error_works.append(workname)
    for w in end_work:
        processes.pop(w)
    time.sleep(10)

print('error works: ', error_works)
