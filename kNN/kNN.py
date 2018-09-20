"""
Created on 2018/9/9
kNN:

Input:

Output:

@Author: Eyre Young
"""
import numpy as np
import operator
import os


def createDataSet():
    """
    创建简单的数据集和标签集

    :return:数据集 标签集
    """
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    """
    kNN分类函数

    :param inX: 用于分类的向量（测试数据）
    :param dataSet:训练数据集features
    :param labels:训练数据集labels
    :param k:最近邻的个数
    :return:分类结果
    """
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteILabel = labels[sortedDistIndicies[i]]
        classCount[voteILabel] = classCount.get(voteILabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    """
    导入训练数据

    :param filename:数据文件路径
    :return:特征值矩阵 标签向量
    """
    fr = open(filename, 'r')
    # 获取数据行数
    numberOfLines = len(fr.readlines())

    # 生成对应空矩阵
    returnMat = np.zeros((numberOfLines, 3))

    classLabelVector = []
    index = 0
    # readlines()函数重新定位到起始点
    fr.seek(0)
    for line in fr.readlines():
        line = line.strip()
        # 用'\t'切分字符串
        listFromLine = line.split('\t')

        # feature数据
        returnMat[index] = listFromLine[0: 3]

        # label数据
        classLabelVector.append(int(listFromLine[-1]))

        index += 1

    return returnMat, classLabelVector


def autoNorm(dataSet):
    """
    归一化特征值

    :param dataSet: 数据集
    :return: 归一化处理后的数据集 归一化处理的范围 最小值
    """
    # 计算每个属性的最大值和最小值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def classifyPerson():
    """

    :return:
    """
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("Percentage of time spent on playing video games ?"))
    ffMiles = float(input("Frequent filer miles earned per years ?"))
    iceCream = float(input("Liters of icecream consumed per year ?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print('You will probably like this person: ', resultList[classifierResult - 1])


def img2vector(filename):
    """
    图像数据转化为向量

    :param filename: 图像文件
    :return: 处理完成的一维矩阵
    """
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect