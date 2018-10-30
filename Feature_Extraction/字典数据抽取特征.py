# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/10/30 2:23 PM'
from sklearn.feature_extraction import  DictVectorizer

def dictvec():
    """
    字典数据特征的抽取
    :return: 
    """
    measurements = [{'city': 'Beijing', 'temperature': 33.},
                    {'city': 'London', 'temperature': 12.},
                    {'city': 'San Fransisco', 'temperature': 18.}]
    dict = DictVectorizer(sparse=False)
    data = dict.fit_transform(measurements)
    print(dict.get_feature_names())
    print(dict.inverse_transform(data))
    print(data)

    return None



if __name__ == "__main__":
    dictvec()