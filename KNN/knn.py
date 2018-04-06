#!/usr/bin/python
# -*- coding: utf8 -*-
from numpy import *
import operator
from os import listdir

def createDataSet():
    group = array([[1.0,1.1], [1.0,1.0], [0,0], [0,0.1]])
    labels = ['A', 'A','B', 'B']
    return group, labels# 返回数据集、标签

def classify0(inX, dataSet, labels, k):
    # 0计算行数，1计算列数
    dataSetsize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetsize,1)) - dataSet
    # 前面用tile
    # 把一行inX变成4行一模一样的（tile有重复的功能，dataSetSize是重复4遍，
    # 后面的1保证重复完了是4行，而不是一行里有四个一样的），
    sqDiffMat = diffMat**2
    # **2代表二次幂,每一行的数都要相加
    sqDistance = sqDiffMat.sum(axis=1)
    # axis=1是行相加，，这样得到了(x1-x2)^2+(y1-y2)^2
    distances = sqDistance**0.5
    sortedDistIndices = distances.argsort()
    # argsort是排序，将元素按照由小到大的顺序返回下标，
    classCount = {}
    # 字典
    for i in range(k):
        # 记录该样本数据所属的类别
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    # 对类别出现的频次进行排序，从高到低
    # operator.itemgetter(1)获取对象第一个域的值
    # reverse
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回出现频次最高的类别
    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()

    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        # \t 表示数据从单元格中截取出来
        listFromLine = line.split('\t')
        # 0,1,2三列的值
        returnMat[index, :] = listFromLine[0:3]
        # 将最后一行的喜欢率加入列表
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector
# 归一化特征值，将特征值转化为0-1之间的数值
def autoNorm(dataSet):
    # 从列中选取最小值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    # dataSet最大值减去最小值/取值范围
    ranges = maxVals - minVals
    #
    normDataSet = zeros(shape(dataSet))
    # 第一行的长度，第一维的长度
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix ('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm (datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount  = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],  normMat[numTestVecs:m,:], datingLabels[numTestVecs:m], 3)
        print("the classifiter came back with: %d, the real answer is: %d  " %(classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
        print("the total error rate is : %f" %(errorCount/float(numTestVecs)))

def classifyPerson():
    resultList = ['not at all', 'in small does', 'in large does']
    percenTats = float(input("percent spent time in play video games ?"))
    ffmiles = float (input (" frequent flier miles in per year? "))
    iceCream = float(input("liters of ice  cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffmiles, percenTats, iceCream ])
    classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
    print('You will probably like this person :', resultList[classifierResult - 1])

def img2vector(filename):
    # 生成1*1024的array
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(0,32):
        lineStr = fr.readline()
        for j in range(0,32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    #list(列表)来记录分类
    hwLabels = []
    #listdir可以列出dir里边的所有文件和目录，但不包括子目录中的内容
    trainingFileList = listdir("trainingDigits")
    #求出文件的长度
    m = len(trainingFileList)
    #生成 m*1024的array，每个文件分配1024个0
    trainingMat = zeros((m, 1024))
    #对每一个file进行循环
    for i in range(m):
        #当前文件（m个文件）
        fileNameStr = trainingFileList[i]
        #将整个文件名用 '.' 分开   文件名格式为 0_0.txt，取第一部分也就是0_0
        fileStr = fileNameStr.split('.')[0]
        #将0_0以 '_' 的形式分开，取第一部分也就是0
        classNumStr = int(fileStr.split('_')[0])
        #将分类的信息加入表中 ，[] 中可以有重复的类别
        hwLabels.append(classNumStr)
        #调用img2vector,将原文件写入trainingMat
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print('the classifier came back with: %d, the real answer is: %d' %(classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print("\n the total umber of errors is: %d" % errorCount)
    print("\n the total error rate is: %f" %(errorCount/float(mTest)))