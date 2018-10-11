# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/5/30 下午8:27'
import threading
import _thread
import time
'''
线程的相关知识
'''

def print_time(thread, delay):
    count = 0
    while count < 5:
        time.sleep (delay)
        count += 1
        print ("%s:%s" % (thread, time.ctime (time.time ())))


try:
    _thread.start_new_thread (print_time, ('thread1', 2))
    _thread.start_new_thread (print_time, ('thread2', 4))
except:
    print ('ERROR, start error!')

while 1:
    pass