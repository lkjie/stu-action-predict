#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'lkjie'

import sys

# 展示变量、清理内存
def showvairbales():
    def sizeof_fmt(num, suffix='B'):
        ''' By Fred Cirera, after https://stackoverflow.com/a/1094933/1870254'''
        for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
            if abs(num) < 1024.0:
                return "%3.1f%s%s" % (num, unit, suffix)
            num /= 1024.0
        return "%.1f%s%s" % (num, 'Yi', suffix)

    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                             key=lambda x: -x[1])[:100]:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))

    # for v in dir():
    #     if re.match('_\d{1,3}', v):
    #         del globals()[v]
    # del globals()['consum_bak']
    import gc
    gc.collect()