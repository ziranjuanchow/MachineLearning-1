# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/12/26 8:13 PM'

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import  train_test_split


# data
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    for i in range(len(data)):
        if data[i,-1] == 0:
            data[i,-1] = -1
    return data[:,:2], data[:,-1]

X, y = create_data()
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)

# plt.scatter(X[:50,0],X[:50,1], label='0')
# plt.scatter(X[50:,0],X[50:,1],label='1')
# plt.legend()
# plt.show()

class SVM:
    def __init__(self,max_iter=100,kernel='linear'):
        """
        :param max_iter: 
        :param kernel: 
        """
        self.max_iter = max_iter
        self._kernel = kernel


    def init_args(self,features,labels):
        """
        :param features: 
        :param labels: 
        :return: 
        """
        self.m,self.n = features.shape
        self.X = features
        self.Y = labels
        self.b = 0.0

        #将Ei保存在一个列表里
        self.alpha = np.ones(self.m)
        self.E = [self._E(i) for i in range(self.m)]
        # 松弛变量
        self.C = 1.0


    def _KKT(self,i):
        """
        :param i: 
        :return: 
        """
        y_g = self._g(i) * self.Y[i]
        if self.alpha[i] == 0:
            return y_g >= 1
        elif 0 < self.alpha[i] < int(self.C):
            return y_g == 1
        else:
            return y_g <= 1


    def _g(self, i):
        """
        g(x)预测值 输入xi(X[i])
        :param i: 
        :return: 
        """
        r = self.b
        for j in range(self.m):
            r += self.alpha[j] * self.Y[j] * self.kernel(self.X[i],self.X[j])

    def kernel(self,x1,x2):
        """
        核函数
        :param x1: 
        :param x2: 
        :return: 
        """
        if self._kernel == 'linear':
            return sum([x1[k] * x2[k] for k in range(self.n)])
        elif self._kernel == 'poly':
            return (sum([x1[k] * x2[k] for k in range(self.n)]) + 1)**2
        return 0


    def _E(self,i):
        """
        E 为g(x)对输入x的预测值和y的差
        :param i: 
        :return: 
        """
        return self._g(i) - self.Y[i]


    def _init_alpha(self):
        """
        :return: 
        """
        # 外层循环遍历所有满足0<a<C的样本点，检验是否满足KKT
        index_list = [i for i in range(self.m) if 0 < self.alpha[i] < int(self.C)]
        # 否则遍历整个训练集
        non_satisfy_list = [i for i in range(self.m) if i not in index_list]
        index_list.extend(non_satisfy_list)

        for i in index_list:
            if self._KKT(i):
                continue

            E1 = self.E[i]
            # 如果E2是正，选择最小的；如果E2是负，选择最大的
            if E1 >= 0:
                j = min(range(self.m),key=lambda x:self.E[x])
            else:
                j = max(range(self.m),key=lambda x:self.E[x])
        return i,j


    def _compare(self, _alpha, L, H):
        """
        :param _alpha: 
        :param L: 
        :param H: 
        :return: 
        """
        if _alpha > H:
            return H
        elif _alpha < L:
            return L
        else:
            return _alpha


    def fit(self, features, labels):
        """
        :param features: 
        :param labels: 
        :return: 
        """
        self.init_args(features, labels)

        for t in range(self.max_iter):
            # train
            i1, i2 = self._init_alpha()

            # 边界
            if  self.Y[i1] == self.Y[i2]:
                L = max(0, self.alpha[i1] + self.alpha[i2] - self.C)
                H = min(self.C, self.alpha[i1] + self.alpha[i2])
            else:
                L = max (0, self.alpha[i2] - self.alpha[i1])
                H = min (self.C, self.C + self.alpha[i2] - self.alpha[i1])
            E1 = self.E[i1]
            E2 = self.E[i2]
            eta = self.kernel(self.X[i1], self.X[i1]) + self.kernel(self.X[i2],self.X[i2]) -\
            2 * self.kernel(self.X[i1],self.X[i2])

            if eta <= 0:
                continue
            alpha2_new_unc = self.alpha[i2] + self.Y[i2] * (E1 -E2)/eta # 书上是 E1-E2
            alpha2_new = self._compare(alpha2_new_unc, L, H)

            alpha1_new = self.alpha[i1] + self.Y[i1] * self.Y[i2] * (self.alpha[i2] - alpha2_new)

            b1_new = -E1 - self.Y[i1] * self.kernel(self.X[i1], self.X[i1]) * (alpha1_new-self.alpha[i1]) - self.Y[i2] * self.kernel(self.X[i2], self.X[i1]) * (alpha2_new-self.alpha[i2])+ self.b
            b2_new = -E2 - self.Y[i1] * self.kernel(self.X[i1], self.X[i2]) * (alpha1_new-self.alpha[i1]) - self.Y[i2] * self.kernel(self.X[i2], self.X[i2]) * (alpha2_new-self.alpha[i2])+ self.b

            if 0 < alpha1_new < int(self.C):
                b_new = b1_new
            elif 0 < alpha2_new < int(self.C):
                b_new = b2_new
            else:
                # 选择中点
                b_new = (b1_new + b2_new) / 2

            # 更新参数
            self.alpha[i1] = alpha1_new
            self.alpha[i2] = alpha2_new
            self.b = b_new

            self.E[i1] = self._E(i1)
            self.E[i2] = self._E(i2)
        return 'train done!'

    def predict(self,data):
        """
        :param data: 
        :return: 
        """
        r = self.b
        for i in range(self.m):
            r += self.alpha[i] * self.Y[i] * self.kernel(data,self.X[i])
        return 1 if r > 0 else -1


    def score(self,X_test,y_test):
        """
        :param X_test: 
        :param y_test: 
        :return: 
        """
        right_count = 0
        for i in range(len(X_test)):
            result = self.predict(X_test[i])
            if result == y_test[i]:
                right_count += 1
        return right_count / len(X_test)


    def _weight(self):
        # Linear Model
        yx = self.Y.reshape(-1,1) * self.X
        self.w = np.dot(yx.T, self.alpha)
        return  self.w

svm = SVM(max_iter=200)
print(svm.fit(X_train,y_train))
score = svm.score(X_test,y_test)
print("SVM 的打分情况 {}".format(score))
