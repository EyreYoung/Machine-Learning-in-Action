"""
Created on 2018/9/10
test:

Input: 

Output: 

@Author: Eyre Young
"""
import kNN
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def classify0Test():
    """
    测试第一个分类器

    :return:
    """
    group, labels = kNN.createDataSet()
    print('group:', group)
    print('labels:', labels)
    print('result:', kNN.classify0([0.1, 0.1], group, labels, 3))


def file2matrixTest():
    returnMat, classLabelVector = kNN.file2matrix("C:\\Users\yangy\PycharmProjects\MLIA\kNN\datingTestSet2.txt")
    print('returnMat:', returnMat)
    print('classLabelVector:', classLabelVector)


def matplotlibTest():
    returnMat, classLabelVector = kNN.file2matrix("C:\\Users\yangy\PycharmProjects\MLIA\kNN\datingTestSet2.txt")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(returnMat[:, 0], returnMat[:, 1], 15.0 * np.array(classLabelVector), 15.0 * np.array(classLabelVector))
    plt.show()


def autoNormTest():
    returnMat, classLabelVector = kNN.file2matrix("C:\\Users\yangy\PycharmProjects\MLIA\kNN\datingTestSet2.txt")
    normMat, ranges, minVals = kNN.autoNorm(returnMat)
    print('normMat:', normMat)
    print('ranges:', ranges)
    print('minVals:', minVals)


def datingClassTest():
    """
    约会网站测试

    :return:
    """
    # 设置测试数据比例
    hoRatio = 0.1

    # 从文件中加载数据
    datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')

    # 归一化数据
    normMat, ranges, minVals = kNN.autoNorm(datingDataMat)

    # m = 数据的行数 即第一维矩阵
    m = normMat.shape[0]

    # 设置测试的样本数量
    numTestVecs = int(m * hoRatio)
    print('numTestVecs = ', numTestVecs)

    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = kNN.classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print('The classifier came back with %d, the real answer is: %d' % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print('The total error rate is %f' % (errorCount / float(numTestVecs)))
    print(errorCount)


def img2vectorTest():
    testVector = kNN.img2vector('testDigits/0_13.txt')
    print(testVector[0, 0 : 32])


def main():
    img2vectorTest()


main()
