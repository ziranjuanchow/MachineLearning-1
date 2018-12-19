# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/12/18 4:59 PM'
import math
from collections import Counter
from random import random
from time import clock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

'''
p=1 曼哈顿距离
p=2 欧式距离
p=inf 闵式距离
'''

def L(x,y,p=2):
    if len(x) == len(y) and len(x) > 1:
        sum = 0
        for i in range(len(x)):
            sum += math.pow(abs(x[i] - y[i]), p)
            return math.pow(sum, 1/p)
    else:
        return 0

x1 = [1, 1]
x2 = [5, 1]
x3 = [4, 4]

for i in range(1,5):
    r = {'1-{}'.format(c):L(x1,c,p=i) for c in [x2,x3]}
    print(min(zip(r.values(),r.keys())))


iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
print(df)

plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()

data = np.array(df.iloc[:100, [0, 1, -1]])
X, y = data[:,:-1], data[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

class KNN:
    def __init__(self,X_train, y_train, n_neighbors=3, p=2):
        '''
        :param X_train: 
        :param y_train: 
        :param n_neighbors: 临近点个数
        :param p: 距离度量
        '''
        self.n = n_neighbors
        self.p = p
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X):
        '''
        :param X: 
        :return: 
        '''
        knn_list = []
        for i in range(self.n):
            dist = np.linalg.norm(X - self.X_train[i], ord=self.p )
            knn_list.append((dist,self.y_train[i]))

        for i in range(self.n, len(self.X_train)):
            max_index = knn_list.index(max(knn_list, key=lambda x:x[0]))
            dist = np.linalg.norm(X - self.X_train[i], ord=self.p)
            if knn_list[max_index][0] > dist:
                knn_list[max_index] = (dist, self.y_train[i])

        knn = [k[-1] for k in knn_list]
        count_paris = Counter(knn)
        max_count = sorted(count_paris, key=lambda x:x)[-1]
        return max_count

    def score(self, X_test, y_test):
        right_count = 0
        n = 10
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if label == y:
                right_count += 1
        return right_count / len(X_test)

clf = KNN(X_train,y_train)
print(clf.score(X_test, y_test))

test_point = [6.0, 3.0]
print('Test Point:{}'.format(clf.predict(test_point)))



plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
plt.plot(test_point[0], test_point[1], 'bo', label='test_point')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()




# 使用scikitlearn模型的方法来进行处理
clf_sk = KNeighborsClassifier()
clf_sk.fit(X_train, y_train)
sk_score = clf_sk.score(X_test,y_test)
print(sk_score)

# KD-Tree的节点的数据结构
class KdNode(object):
    def __init__(self,dom_elt, split, left, right):
        '''
        :param dom_elt: k维向量节点k
        :param split: 分割维度的序号
        :param left: 左子空间构成kd-tree
        :param right: 右子空间构成kd-tree
        '''
        self.dom_elt = dom_elt
        self.split = split
        self.left = left
        self.right = right

class KdTree(object):
    def __init__(self,data):
        k = len(data[0])

        def CreateNode(split, data_set):
            if not data_set:
                return None

            data_set.sort(key=lambda x: x[split])
            split_pos = len(data_set) // 2
            median = data_set[split_pos]
            split_next = (split + 1) % k

            return KdNode(median,split,
                          CreateNode(split_next, data_set[:split_pos]),
                          CreateNode(split_next, data_set[split_pos + 1:]))

        self.root = CreateNode(0, data)

# KdTree的前序遍历
def preorder(root):
    print(root.dom_elt)
    if root.left:
        preorder(root.left)
    if root.right:
        preorder(root.right)


#对构建好的kd树进行搜索，寻找与目标点最近的样本点
from math import sqrt
from collections import namedtuple

result = namedtuple("Result_tuple", "nearest_point nearest_dist nodes_visited")

def find_nearest(tree, point):
    # 数据维度
    k = len(point)

    def travel(kd_node, target, max_dist):
        if kd_node is None:
            # python中用float("inf")和float("-inf")表示正负无穷
            return result([0] * k, float("inf"), 0)
        nodes_visited = 1
        #分割的维度
        s = kd_node.split
        # 分割的 轴
        pivot = kd_node.dom_elt

        if target[s] <= pivot[s]:
            nearer_node = kd_node.left
            further_node = kd_node.right
        else:
            nearer_node = kd_node.right
            further_node = kd_node.left

        temp1 = travel(nearer_node, target, max_dist)

        nearest = temp1.nearest_point
        dist = temp1.nearest_dist

        nodes_visited += temp1.nodes_visited

        if dist < max_dist:
            max_dist = dist

        temp_dist = abs(pivot[s] - target[s])
        if max_dist < temp_dist:
            return result(nearest, dist, nodes_visited)

        # 计算目标点和分割点的欧氏距离
        temp_dist = sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(pivot, target)))
        if temp_dist < dist:
            nearest = pivot
            dist = temp_dist
            max_dist = dist

        # 检查另一个子节点对应的区域是否有更近的节点
        temp2 = travel(further_node, target, max_dist)

        nodes_visited += temp2.nodes_visited
        if temp2.nearest_dist < dist:
            nearest = temp2.nearest_point
            dist = temp2.nearest_dist

        return result(nearest, dist, nodes_visited)
    return travel(tree.root, point, float("inf"))

data = [[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]]
kd = KdTree(data)
res = preorder(kd.root)
print(res)


# 生成一个k维随机向量，每维的值在0-1之间
def random_point(k):
    return [random() for _ in range(k)]

# 产生n个k维随机向量
def random_points(k,n):
    return [random_point(k) for _ in range(n)]

ret = find_nearest(kd, [3,4,5])
print(ret)

# 四十万个三维空间
N = 400000
t0 = clock()
kd2 = KdTree(random_points(3,N))
ret2 = find_nearest(kd2, [0.1,0.5,0.8])
t1 = clock()
print("time:",t1-t0,"s")
print(ret2)