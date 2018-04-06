# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/4/6 上午10:39'


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.externals.six import  StringIO
from sklearn import tree
import pandas as pd
import numpy as np
import pydotplus # 绘制pdf图片

if __name__ == '__main__':
    with open('lenses.txt') as fr:# 加载文件
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]# 一行一行的输出文件
        print(lenses)

    lenses_target = []
    for each in lenses:
        lenses_target.append(each[-1])#
        print(each)

    # print(lenses)

    lenses_labels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lenses_list = []
    lenses_dict = {}
    for each_label in lenses_labels:
        for each in lenses:
            lenses_list.append(each[lenses_labels.index(each_label)])
        lenses_dict[each_label] = lenses_list
        lenses_list = []
    # print(lenses_dict)
    lenses_pd = pd.DataFrame(lenses_dict)


    le = LabelEncoder() # 序列化的作用
    for col in lenses_pd.columns: # lenses_pd中的数据转换为数字代表的值
        lenses_pd[col] = le.fit_transform(lenses_pd[col])

    print(lenses_pd)

    clf = tree.DecisionTreeClassifier(max_depth=4)
    clf = clf.fit(lenses_pd.values.tolist(), lenses_target)

    # dot_data = StringIO()
    # tree.export_graphviz(clf, out_file=dot_data,
    #                      feature_names=lenses_pd.keys(),
    #                      class_names=clf.classes_,
    #                      filled=True,rounded=True,
    #                      special_characters=True
    #                      )
    # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    # graph.write_pdf("tree.pdf")

    print(clf.predict([[1,1,1,0]]))



