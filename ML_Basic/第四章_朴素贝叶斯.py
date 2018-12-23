# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/12/23 3:59 PM'
'''
基于贝叶斯定理与特征条件独立假设的分类方法
模型：
    高斯模型
    多项式模型
    伯努利模型
'''
import math

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


# 伯努利模型和多项式模型
# from sklearn.naive_bayes import BernoulliNB,MultinomialNB


def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length','sepal width','petal length','petal width', 'label']
    data = np.array(df.iloc[:100, :])
    return data[:,:-1], data[:,-1]

X, y = create_data()
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.3)
print(X_test[0],y_test[0])

class NaiveBayes:

    def __init__(self):
        self.model = None

    # 数学期望
    @staticmethod
    def mean(X):
        return sum(X)/float(len(X))

    # 标准差（方差
    def stdev(self, X):
        avg = self.mean(X)
        return math.sqrt(sum([pow(x-avg,2) for x in X]) / float(len(X)))

    # 概率密度函数
    def gaussian_probability(self, x, mean, stdev):
        exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
        return (1/(math.sqrt(2*math.pi) * stdev)) * exponent

    # 处理X_train
    def summarize(self,train_data):
        summaries = [(self.mean(i),self.stdev(i)) for i in zip(*train_data)]
        return summaries

    # 分别求出数学期望和标准差
    def fit(self,X,y):
        labels = list(set(y))
        data = {label:[] for label in labels}
        for f, label in zip(X,y):
            data[label].append(f)
        self.model = {label:self.summarize(value) for label,value in data.items()}
        return 'GaussianNB train done!'

    # 计算概率
    def calculate_probabilities(self, input_data):
        probabilities = {}
        for label, value in self.model.items():
            probabilities[label] = 1
            for i in range(len(value)):
                mean, stdev = value[i]
                probabilities[label] *= self.gaussian_probability(input_data[i], mean, stdev)
        return probabilities

    # 类别
    def predict(self, X_test):
        # 根据倒数第一个数的值进行排序
        label = sorted(self.calculate_probabilities(X_test).items(), key=lambda x: x[-1])[-1][0]
        return label

    # 打分
    def score(self, X_test, y_test):
        right = 0
        for X, y in zip(X_test,y_test):
            label = self.predict(X)
            if label == y:
                right += 1
        return right / float(len(X_test))

if __name__ == "__main__":
    X_data = [4.4,3.2,1.3,0.2]

    # 实现的贝叶斯模型
    model = NaiveBayes()
    model.fit(X_train,y_train)
    print(model.predict(X_data))
    print(model.score(X_test,y_test))

    # sklearn中现有的包
    clf = GaussianNB()
    clf.fit(X_train,y_train)
    print(clf.score(X_test,y_test))
    print(clf.predict([X_data]))
