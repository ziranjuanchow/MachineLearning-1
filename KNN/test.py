import knn
from numpy import *
from matplotlib import pyplot as  plt

group, labels = knn.createDataSet()
print(knn.classify0([0,0], group, labels, 3))
datingDataMat,datingLabels = knn.file2matrix('datingTestSet2.txt')


print(datingDataMat)

# 输出like rate
print(datingLabels[0:20])
# for item in knn.autoNorm(datingDataMat):
    # print(item)

#knn.datingClassTest()
# knn.classifyPerson()
textVector = knn.img2vector('testDigits/0_13.txt')
print(textVector[0, 0:31])
print(textVector[0, 32:63])
# knn.handwritingClassTest()
# fi = plt.figure()
# ax = fi.add_subplot(111)
# ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0*array(datingLabels), 15.0*array(datingLabels))#显示datingDataMat的第一列和第二列的值
# plt.show()


