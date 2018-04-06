# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/4/6 上午11:59'


from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
import pandas as pd




def readData(data):
    train_data = pd.read_csv(data)
    train_data = train_data[['age','income','student','credit','buy']]
    le = LabelEncoder()
    for col in train_data.columns:
        train_data[col] = le.fit_transform(train_data[col])
    print(train_data)
    train_data_type = []
    train_data_type = train_data['buy']
    print(train_data_type)
    train_data.drop(['buy'], axis=1, inplace=True)

    reg = linear_model.BayesianRidge()
    reg.fit(train_data,train_data_type)
    test = [2,1,1,1]
    predict = reg.predict([test])
    print(predict)

    # # print(lenses_dict)
    # lenses_pd = pd.DataFrame (lenses_dict)
    # le = LabelEncoder ()  # 序列化的作用
    # for col in lenses_pd.columns:  # lenses_pd中的数据转换为数字代表的值
    #     lenses_pd[col] = le.fit_transform (lenses_pd[col])
if __name__ == '__main__':
    data = 'data.csv'
    readData(data)
