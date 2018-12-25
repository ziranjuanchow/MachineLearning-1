# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/12/25 11:32 AM'
from math import exp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data,columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length','sepal width','petal length','petal width','label']
    data = np.array(df.iloc[:100,[0,1,-1]])
    return data[:,:2], data[:,-1]

X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

class LogisticRegressionClassfier:
     def __init__(self,max_iter=200,learning_rate=0.01):
         """
         :param max_iter: 
         :param learning_rate: 
         """
         self.max_iter = max_iter
         self.learning_rate = learning_rate

     def sigmoid(self,x):
         """
         :param x: 
         :return: 
         """
         return 1 / (1+exp(-x))

     def data_matrix(self,X):
         """
         :param X: 
         :return: 
         """
         data_mat = []
         for  d in X:
             data_mat.append([1.0, *d])
         return data_mat

     def fit(self,X, y):
         """
         :param X: 
         :param y: 
         :return: 
         """
         data_mat = self.data_matrix(X)
         self.weights = np.zeros((len(data_mat[0]),1),dtype=np.float32)
         for iter_ in range(self.max_iter):
             for i in range(len(X)):
                 result = self.sigmoid(np.dot(data_mat[i], self.weights))
                 error = y[i] - result
                 self.weights += self.learning_rate * error * np.transpose([data_mat[i]])
         print('LogisticRegression Model(learning_rate={},max_iter={})'.format(self.learning_rate,self.max_iter))

     def score(self,X_test,y_test):
         """
         :param X_test: 
         :param y_test: 
         :return: 
         """
         right = 0
         X_test = self.data_matrix(X_test)
         for x, y in zip(X_test,y_test):
             result = np.dot(x,self.weights)
             if (result > 0 and y == 1) or (result < 0 and y == 0):
                 right += 1
         return right / len(X_test)

lr_clf = LogisticRegressionClassfier()
lr_clf.fit(X_train, y_train)
print(lr_clf.score(X_test,y_test))

x_points = np.arange(4,8)
y_ = -(lr_clf.weights[1] * x_points + lr_clf.weights[0]) / lr_clf.weights[2]
plt.plot(x_points, y_)

plt.scatter(X[:50,0],X[:50,1], label='0')
plt.scatter(X[50:,0],X[50:,1], label='1')
plt.legend()
plt.show()

