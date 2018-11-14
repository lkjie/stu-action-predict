#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'liwenjie'

import subprocess
import time


cmd1 = '''
ps aux|grep '11301:11.0.0.13:5900 lkjie@11.0.0.13'
'''
cmd2 = '''
ssh -p 22 -CfNg -L 11301:11.0.0.13:5900 lkjie@11.0.0.13 -o TCPKeepAlive=yes
'''
sub = subprocess.Popen(cmd1, shell=True, cwd="/home/liwenjie", stdout=subprocess.PIPE)
sub.wait()
reslist = sub.stdout.readlines()
nogrepnum = len(list(filter(lambda x:'grep' not in x, reslist)))
rcode = 1
while rcode and nogrepnum < 1:
    sub = subprocess.Popen(['./autossh.sh'], shell=False, cwd="/home/liwenjie")
    rcode = sub.wait()
    print('rcode :{}'.format(rcode))
time.sleep(5*60)