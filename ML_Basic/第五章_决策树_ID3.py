# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/12/23 8:49 PM'
import math
from collections import Counter

import numpy as np
import pandas as pd


class Node:
    def __init__(self,x=None, label=None, y=None, data=None):
        """     
        :param x: 特征
        :param label: 子节点分类依据的特征
        :param y: 类标记（叶子节点才有）
        :param data: 包含数据
        :param child: 子节点
        """
        self.label = label
        self.x = x
        self.child = []
        self.y = y
        self.data = data

    def append(self,node):
        """
        添加子节点
        :param node: 
        :return: 
        """
        self.child.append(node)

    def predict(self,features):
        """
        预测数据所述类
        :param features: 
        :return: 
        """
        if self.y is not None:
            return self.y
        for c in self.child:
            if c.x == features[self.label]:
                return c.predict(features)

def printnode(node,depth=0):
    """
    打印树所有节点
    :param node: 节点
    :param depth: 深度
    :return: 
    """
    if node.label is None:
        print(depth,(node.label,node.x,node.y,len(node.data)))
    else:
        print(depth,(node.label,node.x))
        for c in node.child:
            printnode(c,depth+1)

class DTree:
    def __init__(self,epsilon=0,alpha=0):
        """
        预剪枝、后剪枝参数
        :param epsilon: 
        :param alpha: 
        """
        self.epsilon = epsilon
        self.alpha = alpha
        self.tree = Node()

    def prob(self,datasets):
        """
        求概率
        :param datasets: 
        :return: 
        """
        datalen = len(datasets)
        labelx = set(datasets)
        p = {l:0 for l in labelx}
        for d in datasets:
            p[d] += 1
        for i in p.items():
            p[i[0]] /= datalen
        return p

    def calc_ent(self,datasets):
        """
        求熵
        :param datasets: 
        :return: 
        """
        p = self.prob(datasets)
        ent = sum([-v * math.log(v,2) for v in  p.values()])
        return ent

    def cond_ent(self,datasets,col):
        """
        求条件熵
        :param datasets: 
        :param col: 
        :return: 
        """
        labelx = set(datasets.iloc[col])
        p = {x:[] for x in labelx}
        for i, d in enumerate(datasets.iloc[-1]):
            p[datasets.iloc[col][i]].append(d)
        return sum([self.prob(datasets.iloc[col])[k] * self.calc_ent(p[k]) for k in p.keys()])

    def info_gain_train(self,datasets, datalabels):
        """
        求信息增益{互信息}
        :param datasets: 
        :param datalabels: 
        :return: 
        """
        datasets = datasets.T
        ent = self.calc_ent(datasets.iloc[-1])
        gainmax = {}
        for i in range(len(datasets) - 1):
            cond = self.cond_ent(datasets,i)
            gainmax[ent - cond] = i
        m = max(gainmax.keys())
        return gainmax[m], m

    def train(self,datasets,node):
        """
        训练
        :param datasets: 
        :param node: 
        :return: 
        """
        labely = datasets.columns[-1]
        if len(datasets[labely].value_counts()) == 1:
            node.data = datasets[labely]
            node.y = datasets[labely][0]
            return
        if len(datasets.columns[:-1]) == 0:
            node.data = datasets[labely]
            node.y = datasets[labely].value_counts().index[0]
            return

        gainmaxi, gainmax = self.info_gain_train(datasets,datasets.columns)
        # 如果信息增益（互信息）为零 则表示输入特征x完全相同而标签y相反
        if gainmax <= self.epsilon:
            node.data = datasets[labely]
            node.y = datasets[labely].value_counts().index[0]
            return

        vc = datasets[datasets.columns[gainmaxi]].value_counts()
        for Di in vc.index:
            node.label = gainmaxi
            child = Node(Di)
            node.append(child)
            new_datasets = pd.DataFrame([list(i) for i in datasets.values if i[gainmaxi] == Di], columns=datasets.columns )
            self.train(new_datasets, child)

    def fit(self,datasets):
        """

        :param datasets: 
        :return: 
        """
        self.train(datasets,self.tree)

    def find_leaf(self,node,leaf):
        """
        找到所有的子节点
        :param node: 
        :param leaf: 
        :return: 
        """
        for t in node.child:
            if t.y is not Node:
                leaf.append(t.data)
            else:
                for c in node.child:
                    self.find_leaf(c,leaf)

    def find_father(self,node,errormin):
        """
        找到父节点
        :param node: 
        :param errormin: 
        :return: 
        """
        if node.label is not None:
            cy = [c.y for c in node.child]
            # 全是叶节点
            if None not in  cy:
                childdata = []
                for c in node.child:
                    for d in list(c.data):
                        childdata.append(d)
                childcounter = Counter(childdata)

                old_child = node.child
                old_label = node.label
                old_y = node.y
                old_data = node.data

                # 剪枝
                node.label = None
                node.y = childcounter.most_common(1)[0][0]
                node.data = childdata

                error = self.c_error()
                # 剪枝前后损失比较
                if error <= errormin:
                    errormin = error
                    return 1
                else:
                    # 剪枝效果不好，复原
                    node.child = old_child
                    node.label = old_label
                    node.y = old_y
                    node.data = old_data
            else:
                re = 0
                i = 0
                while i < len(node.child):
                    # 如果剪过枝，则其父节点要重新检测
                    if_re = self.find_father(node.child[i],errormin)
                    if if_re == 1:
                        re = 1
                    elif if_re == 2:
                        i -= 1
                    i += 1
                if re:
                    return 2
        return 0

    def c_error(self):
        """
        求C(T)        
        :return: 
        """
        leaf = []
        self.find_leaf(self.tree, leaf)
        leafnum = [len(l) for l in leaf]
        ent = [self.calc_ent(l) for l in  leaf]
        print("Ent:{}".format(ent))
        error = self.alpha * len(leafnum)
        for l, e in zip(leafnum, ent):
            error += l * e
        print("C(T): {}".format(error))
        return error

    def cut(self,alpha=0):
        """
        剪枝
        :param alpha: 
        :return: 
        """
        if alpha:
            self.alpha = alpha
        errormin = self.c_error()
        self.find_father(self.tree, errormin)

datasets = np.array([['青年', '否', '否', '一般', '否'],
                     ['青年', '否', '否', '好', '否'],
                     ['青年', '是', '否', '好', '是'],
                     ['青年', '是', '是', '一般', '是'],
                     ['青年', '否', '否', '一般', '否'],
                     ['中年', '否', '否', '一般', '否'],
                     ['中年', '否', '否', '好', '否'],
                     ['中年', '是', '是', '好', '是'],
                     ['中年', '否', '是', '非常好', '是'],
                     ['中年', '否', '是', '非常好', '是'],
                     ['老年', '否', '是', '非常好', '是'],
                     ['老年', '否', '是', '好', '是'],
                     ['老年', '是', '否', '好', '是'],
                     ['老年', '是', '否', '非常好', '是'],
                     ['老年', '否', '否', '一般', '否'],
                     ['青年', '否', '否', '一般', '是']])

datalabels = np.array(['年龄','有工作','有自己的房子','信贷情况','类别'])
train_data = pd.DataFrame(datasets, columns=datalabels)
test_data = ['老年','否','否','一般']

dt = DTree(epsilon=0)
dt.fit(train_data)

print('DTree:')
printnode(dt.tree)
y = dt.tree.predict(test_data)
print("result: {}".format(y))

# 可修改正则化参数alpha查看后剪枝效果
dt.cut(alpha=0.5)
print("DTree:")
printnode(dt.tree)
y = dt.tree.predict(test_data)
print('result: {}'.format(y))
