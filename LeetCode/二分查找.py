# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/10/14 8:40 PM'
'''
给定一个数，从有序列表中找到我们想要的那个数
'''

def search(sequence, number, lower=0, upper=None):
    if upper is None:
        upper = len(sequence)-1
    if lower == upper:
        assert number == sequence[upper]
        return upper
    else:
        middle = (lower + upper) // 2
        if number > sequence[middle]:
            return search(sequence, number, middle + 1, upper)
        else:
            return search(sequence, number, lower, middle)
seq = [34, 67, 8, 123,4,100,95]
seq.sort()
print(seq)
print(search(seq, 34))
