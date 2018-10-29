"""
Created on 2018/10/29
NaiveBayes: 朴素贝叶斯

屏蔽社区留言板的侮辱性言论

Input: 

Output: 

@Author: Eyre Young
"""
def loadDataSet():
    """
    创建数据集

    :return: 单词列表postingList, 所属类别classVec
    """
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],  # [0,0,1,1,1......]
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec


def createVocabList(dataSet):
    """
    获取所有单词的集合

    :param dataSet: 数据集
    :return: 所以单词集合(不重复)
    """
    vocabSet = set([]) #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #|用来求并集
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    """
    遍历查看单词是否出现

    :param vocabList: 所有单词集合列表
    :param inputSet: 输入数据集
    :return: 匹配列表
    """
    # 创建一个与词汇表等长的向量，元素都置为1
    returnVec = [0] * len(vocabList)

    # 遍历文档中所有单词，出现词汇表中的单词，将输出向量对应值设为1
    for word in vocabList:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec


