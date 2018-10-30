# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/10/30 6:48 PM'
from sklearn.datasets import load_iris, load_boston

'''
分类的数据集
'''
# 加载数据
li = load_iris()

# print("获取值")
# print(li.data)
# print("目标值")
# print(li.target)
# # 数据的描述
# print(li.DESCR)

# 划分训练集和测试集 需要注意返回值  训练集train  x_train y_train   测试集test x_test y_test
# 默认是乱序的输出
# x_train, x_test, y_train, y_test = train_test_split(li.data,li.target,test_size=0.25)
# print("训练集特征值和目标值：",x_train,y_train)
# print("测试集特征值和目标值：",x_test,y_test)
# news = fetch_20newsgroups(data_home="/Users/liudong/Desktop",subset='all')
#
# print(news.data)
# print(news.target)

'''
回归类型的数据 目标值是离散型的数值
'''
# 波士顿房价
lb = load_boston()

print(lb.data)
print(lb.target)
print(lb.DESCR)


