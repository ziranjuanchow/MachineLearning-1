# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/5/30 下午8:38'
import sys
'''
迭代器 iter
'''
list = [1, 2, 3, 4, 5]
it = iter(list)
for x in it:
    print(x, end=" ")
print('=='*10)


'''
生成器 yield
'''
def fibonacci(n): # 生成器函数 - 斐波那契
    a, b, counter = 0, 1, 0
    while True:
        if (counter > n):
            return
        yield a  # a的每次的值进行保存，下次调用时使用当前状态的值
        a, b = b, a + b
        counter += 1
f = fibonacci(10) # f 是一个迭代器，由生成器返回生成

while True:
    try:
        print(next(f), end=" ")
    except StopIteration:
        sys.exit()