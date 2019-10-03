# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 20:18:52 2019

@author: Administrator
"""

import numpy as np
import re
#定义创建列表函数
"""
createVocabList()函数会创建一个包含在所有文档中出现的不重复词的列表

"""
def createVocabList(dataSet):
    #创建一个空集
    vocabSet = set([])  
    for document in dataSet:
        #再创建一个空集后，将每篇文档返回的新词集合添加到该集合中，再求两个集合的并集
        vocabSet = vocabSet | set(document) 
    return list(vocabSet)
#定义词集模型函数（set-of-words）
"""
该函数输入参数为词汇表及某个文档，输出的是文档向量，向量的每一个元素为1或者0，分别
表示词汇表中的单词在输入文档中是否出现

"""
def setOfWords2Vec(vocabList, inputSet):
    #函数首先创建一个和词汇表等长的向量，并将其元素都设置为0
    returnVec = [0]*len(vocabList)
    #接着，遍历文档中的所有单词，如果出现了词汇表中的单词，则将输出的文档向量中的对应值设为1。
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec   
#定义词带模型函数（bag-of-words）
"""
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec
""" 
#定义朴素贝叶斯分类器训练函数
"""
函数说明:朴素贝叶斯分类器训练函数
trainMatrix--训练文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵；trainCategory--训练类别标签向量
p1Vect--标记为1的类条件概率数组；p0Vect--标记为0的类条件概率数组；pAbusive是标记为1类的先验概率
"""
def trainNB(trainMatrix, trainCategory):
    #计算训练的文档数目
    numTrainDocs = len(trainMatrix)
    #计算每篇文档的词条数 相当于特征数
    numWords = len(trainMatrix[0])
    #标记为1类的先验概率
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    """
    创建numpy数组初始化为1，拉普拉斯平滑。
    创建numpy.zeros数组,词条出现数初始化为0。分母初始化为2
    
    """
    p0Num = np.ones(numWords); p1Num = np.ones(numWords)      
    p0Denom = 2.0; p1Denom = 2.0  
    #计算类条件概率
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]#矩阵相加
            p1Denom += 1#表示类1文档个数
           # p1Denom += sum(trainMatrix[i])#有误
        else:
            p0Num += trainMatrix[i]
            p0Denom += 1#表示类0文档个数
            #p0Denom += sum(trainMatrix[i])#有误
    #由于大部分因子都非常小，防止数值下溢得不到正确答案。于是加log计算，可以使得答案不会过小。
    p1Vect = np.log(p1Num/p1Denom)          #change to np.log() 类1 为1 的条件概率
    p0Vect = np.log(p0Num/p0Denom)          #change to np.log() 类0 为0 的条件概率  和后面的计算有一些联系
    return p0Vect, p1Vect, pAbusive   
#定义朴素贝叶斯分类器预测函数 指出该分类器有误？确实是，没有计算类1下为0的条件概率，or 类0下为1的条件概率 https://blog.csdn.net/lming_08/article/details/37542331
"""#vec2Classify 为0 1 组成的特征向量
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)    #element-wise mult 类1发生的时，词特征向量为1的条件概率求和 sum(vec2Classify * p1Vec)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)#类0发生的时，词特征向量为1的条件概率求和
    if p1 > p0:
        return 1
    else:
        return 0
"""

#函数说明:朴素贝叶斯分类器分类函数
#vec2Classify--待分类的词条数组; p1Vec--标记为类1的类条件概率数组; p0Vec--标记为类0的类条件概率数组; pClass1--标记为1类的先验概率
"""
博客
https://blog.csdn.net/qq_27009517/article/details/80044431
https://blog.csdn.net/lming_08/article/details/37542331
"""
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """
    1.计算待分类词条数组为1类的概率
    """
    #寻找vec2Classify测试数组中，元素为0时对应的索引值
    index = np.where(vec2Classify==0)#返回一串索引值，相等为true 否则为false 只返回false索引
    #遍历元素为0时的索引值，并从p1Vec--1类的条件概率数组中取出对应索引的数值，并存储成列表的形式（p1Vec0=[]）
    p1Vec0=[]
    for i in index:#index为tuple 取i=0 的tuple 只执行一次
        for m in i:
            p1Vec0.append(p1Vec[m])
    #所有P(vec2Classify=0|1)组成的列表
        x0=np.ones(len(p1Vec0))-p1Vec0##？和训练过程得到的p1Vec有关，p1Vec它表示类1下为1的条件概率
    #寻找vec2Classify测试数组中，元素为1时对应的索引值
    index1= np.where(vec2Classify==1)
    #遍历元素为1时的索引值，并从p1Vec--1类的条件概率数组中取出对应索引的数值，并存储成列表的形式（p1Vec1=[]）
    p1Vec1=[]
    for i in index1:
        for m in i:
            p1Vec1.append(p1Vec[m])
    #所有P(vec2Classify=1|1)组成的列表
    x1=p1Vec1      
    ##对应元素相乘。logA * B = logA + logB，所以这里加上log(pClass1)
    p1 = sum(x0)+sum(x1) +  np.log(pClass1)        
    """
    2.计算待分类词条数组为0类的概率
    """
    
    #寻找vec2Classify测试数组中，元素为0时对应的索引值
    index2 = np.where(vec2Classify==0)
    #遍历元素为0时的索引值，并从p0Vec--0类的条件概率数组中取出对应索引的数值，并存储成列表的形式（p0Vec0=[]）
    p0Vec0=[]
    for i in index2:
        for m in i:
            p0Vec0.append(p0Vec[m])
    #所有P(vec2Classify=0|0)组成的列表
    w0=np.ones(len(p0Vec0))-p0Vec0
    #寻找vec2Classify测试数组中，元素为1时对应的索引值
    index3= np.where(vec2Classify==1)
    #遍历元素为1时的索引值，并从p0Vec--0类的条件概率数组中取出对应索引的数值，并存储成列表的形式（p0Vec1=[]）
    p0Vec1=[]
    for i in index3:
        for m in i:
            p0Vec1.append(p0Vec[m])
    #所有P(vec2Classify=1|0)组成的列表
    w1=p0Vec1
    ##对应元素相乘。logA * B = logA + logB，所以这里加上log(pClass1)
    p0 = sum(w0)+sum(w1) +  np.log(1.0 - pClass1)
    
    if p1 > p0:
        return 1
    else:
        return 0
#使用朴素贝叶斯过滤垃圾邮件
"""
书本中4.6.1节 准备数据，切分文本部分写的很清晰。
"""
#将一个大字符串解析为字符列表。input is big string, #output is word list
def textParse(bigString):    
    import re
    listOfTokens = re.split(r'\W+', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] #找出长度大于两个字母的单词

def spamTest():
    docList = []; classList = []; fullText = []
    #遍历25个txt文件
    for i in range(1, 26):
        #读取每个垃圾邮件，大字符串转换成字符列表。
        wordList = textParse(open('email/spam/%d.txt' % i, encoding="ISO-8859-1").read())
        docList.append(wordList)#不展开列表
        fullText.extend(wordList)#展开列表
        #标记垃圾邮件，1表示垃圾邮件
        classList.append(1)
        #读取每个非垃圾邮件，字符串转换为字符列表
        wordList = textParse(open('email/ham/%d.txt' % i, encoding="ISO-8859-1").read())
        docList.append(wordList)
        fullText.extend(wordList)
        #标记每个非垃圾邮件，0表示非垃圾邮件
        classList.append(0)
    #创建词汇表，不重复
    vocabList = createVocabList(docList)
    #创建存储训练集的索引值的列表
    trainingSet =list(range(50)); 
    #创建存储测试集的索引值的列表
    testSet= [] 
    #从50个邮件中，随机挑选出40个作为训练集，10个作为测试集
    for i in range(10):
        #随机选取索引值
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        #添加测试集的索引值
        testSet.append(trainingSet[randIndex])
        #在训练集的列表中删除添加到测试集的索引值
        del(list(trainingSet)[randIndex])
    #创建训练集矩阵和训练集类别标签向量
    trainMat = []; 
    trainClasses = []
    #遍历训练集，目前只有40个训练集
    for docIndex in trainingSet:
        #将生成的词集模型添加到训练矩阵中
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        #将类别标签添加到训练集的类别标签向量中
        trainClasses.append(classList[docIndex])
    """
    训练朴素贝叶斯模型
    """
    #训练朴素贝叶斯模型
    p0V, p1V, pSpam = trainNB(np.array(trainMat), np.array(trainClasses))
    #错误分类计数
    errorCount = 0
    #遍历测试集
    for docIndex in testSet:    
        #测试集的词集模型
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error", docList[docIndex])
    print('the error rate is: ', float(errorCount)/len(testSet))

#预测
spamTest()       