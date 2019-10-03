# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 14:55:39 2019

@author: Administrator
"""
from math import log
import operator
import treePlotter as tP
import trees as tr_f
#熵的定义
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    #统计当前数据集的类别标签出现的次数，例如两个类别 类别1出现的次数为x1 类别2出现的次数为x2 总数=x1+x2
    for featVec in dataSet: #the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2) #log base 2
    return shannonEnt
#划分数据集: 按照给定特征划分数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

#选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)#计算原始数据的香农熵
    bestInfoGain = 0.0; bestFeature = -1
    # 依据特征数选择最佳的划分特征
    for i in range(numFeatures):        #iterate over all the features
        featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
        #创建此特征取值的不重复类型
        uniqueVals = set(featList)       #get a set of unique values
        newEntropy = 0.0
        #计算特征下（列）对应的经验条件熵
        for value in uniqueVals:
            #找出第i个特征下对应值的数据样本个数
            subDataSet = splitDataSet(dataSet, i, value)#划分数据集: 按照给定特征划分数据集
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)#计算熵
        #计算信息增益
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
    return bestFeature
#多数表决法决定该叶子节点的分类
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)#降序
    return sortedClassCount[0][0]    
#递归构建决策树
def createTree(dataSet, labels):
    # classList 表示类别
    classList = [example[-1] for example in dataSet]#把dataset中的每一行放入example，然后找到相应的example[-1] 组成列表
   # 参考https://blog.csdn.net/jiangsujiangjiang/article/details/84313227
    #https://blog.csdn.net/weixin_41580067/article/details/82888699
    #两种特殊情况1、最终只剩下一个样本 2、只剩下一个特征
    if classList.count(classList[0]) == len(classList):
        return classList[0]#stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
        return majorityCnt(classList)#多数表决法决定该叶子节点的分类
    #寻找最佳的特征 bestFeat表示索引
    bestFeat = chooseBestFeatureToSplit(dataSet)#选择最好的数据集划分方式
    bestFeatLabel = labels[bestFeat]#找出特征作为根节点
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])#删除此特征
    
    #特征下对应的值
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)#找出不重复的特征值（该特征下可能对应多个特征值）
    for value in uniqueVals:
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree
#使用决策树执行分类
    #解析https://www.cnblogs.com/wyuzl/p/7700872.html
    #依据类别标签和待测数据的的值，找到对应的树子结点，递归的查找。并返回最终分类的值
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree)[0]#当前树的根节点的特征名称 
    secondDict = inputTree[firstStr]#根节点的所有子节点
    featIndex = featLabels.index(firstStr)#找到根节点特征对应的下标  
    key = testVec[featIndex] #找出待测数据的特征值  
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):#判断valueOfFeat是否是dict类型，若是dict类型则说明不是叶结点（叶结点为str）
        classLabel = classify(valueOfFeat, featLabels, testVec)#递归的进入下一层结点
    else: classLabel = valueOfFeat#如果是叶结点，则确定待测数据的分类
    return classLabel    


# main
  #读取眼镜数据并构建树
fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
lensesTree = createTree(lenses,lensesLabels)
print(lensesTree)

# plot tree
tP.createPlot(lensesTree)
#对新数据进行分类
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
testVec=['young','hyper','yes','normal']
result=classify(lensesTree,lensesLabels, testVec)
print(result)

#存储构建的树并加载树
tr_f.storeTree(lensesTree,'ClassfyTree_lenses.txt')
load_tree=tr_f.grabTree('ClassfyTree_lenses.txt')
print(load_tree)
# 原始数据集分类
#lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
#classify(lensesTree, lensesLabels, lenses[0][:-1])
#
#preds = []
#for i in range(len(lenses)):
#    pred = classify(lensesTree, lensesLabels, lenses[i][:-1])
#    preds.append(pred)
#print(preds)