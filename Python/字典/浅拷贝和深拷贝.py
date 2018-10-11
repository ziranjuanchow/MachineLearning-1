# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/8/15 下午8:47'



# 浅拷贝的例子
'''
dict中有copy的方法，属于浅复制
当替换副本中的值的时候，原件不受影响
如果修改副本中的值的（就地修改而不是替换的时候），原件也将发生改变
'''
x = {'username':'admin','machines':['foo','bar','baz']}
y = x.copy()
y['username'] = 'ld'
y['machines'].remove('bar')
print(y)
print(x)

# 深拷贝的例子
# 导入深拷贝
from copy import deepcopy
d = {}
d['names'] = ['Alfred','Bertrand']
c = d.copy()
dc = deepcopy(d)
d['names'].append('Clive')
print(c)
print(dc)

