# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/10/30 2:38 PM'

'''
文本分类 情感分析
特征抽取：  CountVectorizer(对于单个字母不进行统计 因为单个字母无法代表主题)
           tf-idf   主要用于文本分类  计算方式：tf*idf 重要性
           term frequency :词的频率
           idf：逆文档频率 inverse document frequency  log(总文档数量/该词出现的文档数) 输入的数值越小 结果越小
'''
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def countvec():
    """
    对文本进行特征值化 对句子进行处理
    不支持分词的效果
    :return: 
    """
    cv = CountVectorizer()
    data = cv.fit_transform(["人生 苦短， 我 用Python ","人生漫长， 我用哈哈 "])
    # 统计所有文章中出现的次 出现多次只当做一次进行统计
    print(cv.get_feature_names())
    # 将每一篇文章按照出现的5个词 统计每个词的出现次数
    print(data.toarray())

    return None

def jieba_method():
    """
    中文进行分词处理
    :return: None
    """
    con1 = jieba.cut("人生苦短,我用Python 人生漫长，我用哈哈 ")
    con2 = jieba.cut("几年的天到货打号机，大家好复活甲三")
    con3 = jieba.cut("多久啊空间，都会撒娇和。都认为今日")

    #转换成列表
    content1 = list(con1)
    content2 = list(con2)
    content3 = list(con3)
    # 把列表转化为字符串
    c1 = ' '.join(content1)
    c2 = ' '.join(content2)
    c3 = ' '.join(content3)
    print(c1,c2,c3)
    cv = CountVectorizer()
    data  = cv.fit_transform([c1,c2,c3])
    print(cv.get_feature_names())
    print(data.toarray())

def tfidfvec():
    """
    中文进行分词处理
    :return: None
    """
    con1 = jieba.cut("人生苦短,我用Python 人生漫长，我用哈哈 ")
    con2 = jieba.cut("几年的天到货打号机，大家好复活甲三")
    con3 = jieba.cut("多久啊空间，都会撒娇和。都认为今日")

    #转换成列表
    content1 = list(con1)
    content2 = list(con2)
    content3 = list(con3)
    # 把列表转化为字符串
    c1 = ' '.join(content1)
    c2 = ' '.join(content2)
    c3 = ' '.join(content3)
    print(c1,c2,c3)
    tf = TfidfVectorizer()
    data  = tf.fit_transform([c1,c2,c3])
    print(tf.get_feature_names())
    # 输出结果中的数值代表着重要性
    print(data.toarray())


if __name__ == "__main__":
    # countvec()
    # jieba_method()
    tfidfvec()