"""
Created on 2018/9/18
DecisionTree:

Input:

Output:

@Author: Eyre Young
"""
from math import log


def createDataSet():
    """
    创建数据集

    :return:数据集 对应的标签
    """
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]

    # dataSet中两个特征的含义
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def calcShannonEnt(dataSet):
    """
    计算数据集的香农熵

    :param dataSet: 数据集
    :return: 每组feature下的某个分类的香农熵的信息期望
    """
    # 求list的长度，即计算参与训练的数据量
    numEntries = len(dataSet)

    # 计算分类标签label的出现次数
    labelCounts = {}
    for featVec in dataSet:
        # 存储当前实例的标签
        currentLabel = featVec[-1]

        # 为所有可能的分类创建字典
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    # 计算label标签的香农熵
    shannonEnt = 0.0
    for key in labelCounts:
        # 使用所有类标签的发生频率计算类别出现的概率
        prob = float(labelCounts[key]) / numEntries

        # 计算香农熵
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt
