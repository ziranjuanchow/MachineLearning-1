# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/12/18 3:06 PM'
'''
随机梯度下降算法  SGD 

'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# load data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target

df.columns = ['sepal length', 'sepal width','petal length', 'petal width','label']
print(df.label.value_counts())

plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(['0','1'])
plt.show()

data = np.array(df.iloc[:100, [0,1,-1]])
X, y = data[:,:-1], data[:,-1]

y = np.array([i if i == 1 else -1 for i in y])

# 数据线性可分，二分类数据 一元一次线性方程
class Model:
    def __init__(self):
        self.w = np.ones(len(data[0])-1, dtype=np.float32)
        self.b = 0
        self.l_rate = 0.1

    def sign(self, x,w,b):
        y = np.dot(x,w) + b;
        return y

    #SGD
    def fit(self,X_train, y_train):
        is_wrong = False
        while not is_wrong:
            wrong_count = 0
            for d in range(len(X_train)):
                X = X_train[d]
                y = y_train[d]
                if y * self.sign(X, self.w, self.b) <= 0:
                    self.w = self.w + self.l_rate*np.dot(y,X)
                    self.b = self.b + self.l_rate*y
                    wrong_count += 1
                if wrong_count == 0:
                    is_wrong = True
            if wrong_count == 0:
                is_wrong = True
        return 'Perceptron Model!'

perceptron = Model()
print(perceptron.fit(X,y))
x_points = np.linspace(4,7,10)
y_ = -(perceptron.w[0]*x_points + perceptron.b)/perceptron.w[1]
plt.plot(x_points,y_)

plt.plot(data[:50,0], data[:50,1], 'bo', color='blue', label='0')
plt.plot(data[50:100,0], data[50:100,1], 'bo', color='orange', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()

# perceptron 感知器
from sklearn.linear_model import Perceptron

clf = Perceptron(fit_intercept=False, max_iter=1000, shuffle=False)
result = clf.fit(X,y)
print(result)

# weight to the feature
print(clf.coef_)

# 截距
print(clf.intercept_)


x_ponits = np.arange(4, 8)
y_ = -(clf.coef_[0][0]*x_ponits + clf.intercept_)/clf.coef_[0][1]
plt.plot(x_ponits, y_)

plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()

