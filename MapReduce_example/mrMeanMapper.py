# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/5/18 上午11:24'
import  sys
from numpy import mat, mean, power

def read_input(file):
    for line in file:
        yield line.rstrip()

input = read_input(sys.stdin)
input = [float(line) for line in input]
numInput = len(input)
input = mat(input)
sqInput = power(input, 2)
print('%d\t%f\t%f' %(numInput, mean(input), mean(sqInput)))
print(sys.stderr, 'report: still alive!')