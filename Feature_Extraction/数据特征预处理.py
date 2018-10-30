# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/10/30 3:20 PM'
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler, Imputer

'''
数据中异常点较多会出现什么影响？
  异常点较多会出现数据出错比较多。比如最大值和最小值出现异常 会对归一化的结果造成影响
  
'''
'''
归一化：
归一化处理 对数据进行缩放
    x' = (x -min)/(max-min)
    归一化的原因：多个特征同等重要的时候，需要进行归一化
                避免某一个数值比较大的特征，对最终的结果的影响比较大
'''
def MinMax():
    '''           
    :return: None
    '''
    mm = MinMaxScaler()
    data = mm.fit_transform([[10,27,36,47,88],[12,34,55,67,87],[12,32,15,67,35]])
    print(data)
    return None
'''
标准化：通过对原始数据进行变换把数据变换到均值为0 标准差为1的范围内
      结果中出现异常点，少量的异常点对于平均值的影响不大，从而方差改变的比较小
      结果受异常点的影响比较小
      适合有异常值出现的数据
'''
def standardscaler():
    '''
    标准化缩放
    :return: 
    '''
    ss = StandardScaler()
    data = ss.fit_transform ([[1., -1., 3.], [2., 4., 2.], [4., 6., -1.]])
    print (data)
    return None

def im():
    '''
    缺失值处理
    :return: 
    '''
    im =  Imputer(missing_values='NaN',strategy='mean',axis=0)
    data = im.fit_transform([[1,2],[np.nan,3],[6,7]])
    print(data)
    return None

'''
数据降维：把特征的数量进行减少
'''


def var():
    """
    特征选择  删除低方差的特征  过滤式特征选择
    :return: 
    """
    var = VarianceThreshold(threshold=1.0)
    data = var.fit_transform([[0,2,0,3],[0,1,4,3],[0,1,1,3]])

    print(data)
    return None

def pca():
    '''
    主成分分析进行特征降维
    :return: 
    '''
    pca = PCA(n_components=0.9)# 代表是原有数据的90%的数据
    data = pca.fit_transform([[2,8,4,5],[6,3,0,8],[5,4,9,1]])
    print(data)

    return None

if __name__ == "__main__":
    # MinMax()
    # standardscaler()
    # im()
    # var()
    pca()