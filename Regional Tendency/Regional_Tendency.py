# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 17:23:41 2019

@author: Administrator
"""

'''
Created on Oct 19, 2010

@author: Peter
'''
from numpy import *
import numpy as np
import bayes as ba

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help','my','dog', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec
"""
createVocabList()函数会创建一个包含在所有文档中出现的不重复词的列表

"""                 
def createVocabList(dataSet):
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)

def bagOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else: print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

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
"""
函数说明:朴素贝叶斯分类器预测函数
Parameters:
    vec2Classify - 待分类词向量
    p0Vec -  类0的条件概率
    p1Vec -  类1的条件概率
    pClass1 -  先验概率）
Returns:
    returnVec - 输入文档对应的词向量（词频）
Author:
    heda3
""" #有误
# =============================================================================
# def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
#     p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise mult
#     p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
#     if p1 > p0:
#         return 1
#     else: 
#         return 0
# =============================================================================
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

"""
函数说明:词袋模型，依据输入文档是否在词列表中，相应的值累加次数

Parameters:
    vocabList - 所有的词列表（所有文档的并集）
    inputSet -  输入文档（词列表）
Returns:
    returnVec - 输入文档对应的词向量（词频）
Author:
    heda3
Blog:
    https://blog.csdn.net/heda3
Modify:
    2019-10-01
"""    
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def testingNB():
    print('*** load dataset for training ***')
    listOPosts,listClasses = loadDataSet()
    print('listOPost:\n',listOPosts)
    print('listClasses:\n',listClasses)
    print('\n*** create Vocab List ***')
    myVocabList = createVocabList(listOPosts)
    print('myVocabList:\n',myVocabList)
    print('\n*** Vocab show in post Vector Matrix ***')
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(bagOfWords2Vec(myVocabList, postinDoc))
    print('train matrix:',trainMat)
    print('\n*** train P0V p1V pAb ***')
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    print('p0V:\n',p0V)
    print('p1V:\n',p1V)
    print('pAb:\n',pAb)
    print('\n*** classify ***')
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(bagOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(bagOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))

# 英文的分词
def textParse(bigString):    #input is big string, #output is word list
    import re
    #listOfTokens = re.split(r'\W*', bigString)#按照一个字母切分了
    listOfTokens = re.split(r'\W+', bigString)#https://blog.csdn.net/weixin_43744799/article/details/86087542
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] 
    

def spamTest():
    docList=[]; classList = []; fullText =[]
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary
    trainingSet = range(50); testSet=[]           #create test set
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print ("classification error",docList[docIndex])
    print('the error rate is: ',float(errorCount)/len(testSet))
    #return vocabList,fullText

def calcMostFreq(vocabList,fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True) 
    return sortedFreq[:30]       

def stopWords():
    import re
    wordList =  open('./stopwords_en.txt').read() # see http://www.ranks.nl/stopwords
    listOfTokens = re.split(r'\W+', wordList)
    return [tok.lower() for tok in listOfTokens] 
    print('read stop word from \'stopword.txt\':',listOfTokens)
    return listOfTokens

"""
函数说明:根据feature_words将文本向量化

Parameters:
    feed1 - RSS源1
    feed0 - RSS源2
Returns:
    vocabList -不重复的词列表(去掉停用词) 
    p0V - 类0的条件概率
    p1V - 类1的条件概率
Author:
    heda3
Blog:
    https://blog.csdn.net/heda3
Modify:
    2019-09-30
"""
def localWords(feed1,feed0):
    import feedparser
    #引入RSS源，并提取相应的词汇-转换为词列表
    docList=[]; classList = []; fullText =[]
    print('feed1 entries length: ', len(feed1['entries']), '\nfeed0 entries length: ', len(feed0['entries']))
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    print('\nmin Length: ', minLen)
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])#英文分词
        print('\nfeed1\'s entries[',i,']\'s summary - ','parse text:\n',wordList)
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) #NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        print('\nfeed0\'s entries[',i,']\'s summary - ','parse text:\n',wordList)
        docList.append(wordList)
        fullText.extend(wordList)#展开
        classList.append(0)
    #创建不重复词的列表
    vocabList = createVocabList(docList)#create vocabulary
    print('\nVocabList is ',vocabList)
   
    #调用停词列表，移除不重复词列表中包含停词列表中的元素
    print('\nRemove Stop Word:')
    stopWordList = stopWords()
    for stopWord in stopWordList:
        if stopWord in vocabList:
            vocabList.remove(stopWord)#去掉停止词后的词列表
            print('Removed: ',stopWord)
    #计算最高频次的词--从不重复的词列表中删除掉高频词 这个和前面的停用词表的重复了
    #vocabList:不重复的词向量 fullText：所有的词展开
    top30Words =  ba.calcMostFreq(vocabList,fullText)   #remove top 30 words
    print('\nTop 30 words: ', top30Words)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
            print('\nRemoved: ',pairW[0])
    #训练集和测试集划分 并将输入词列表转换为词向量
    trainingSet =list(range(2*minLen));#在python3中range返回的是一个range对象，故转换为列表形式
    testSet=[]           #create test set
    print('\n\nBegin to create a test set: \ntrainingSet:',trainingSet,'\ntestSet',testSet)
    for i in range(5):
        randIndex = int(random.uniform(0,len(trainingSet)))#随机选取在0-n之间的实数
        #从trainingSet中随机找出索引值，并将它放置在测试集里，同时删除测试集的值
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    print('random select 5 sets as the testSet:\ntrainingSet:',trainingSet,'\ntestSet',testSet)
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        #词列表转换为词向量
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    print('\ntrainMat length:',len(trainMat))
    print('\ntrainClasses',trainClasses)
    print('\n\ntrainNB0:')
    #模型训练
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    #print '\np0V:',p0V,'\np1V',p1V,'\npSpam',pSpam
    ##返回 条件概率p0V  p1V，以及标记为1类的先验概率
    #模型测试
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        #词列表转换为词向量
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        classifiedClass = classifyNB(array(wordVector),p0V,p1V,pSpam)
        originalClass = classList[docIndex]
        result =  classifiedClass != originalClass
        if result:
            errorCount += 1
        print('\n',docList[docIndex],'\nis classified as: ',classifiedClass,', while the original class is: ',originalClass,'. --',not result)
    print('\nthe error rate is: ',float(errorCount)/len(testSet))
    return vocabList,p0V,p1V

def testRSS():
    import feedparser
    ny=feedparser.parse('http://www.nasa.gov/rss/dyn/image_of_the_day.rss')#NASA
    sf=feedparser.parse('http://sports.yahoo.com/nba/teams/hou/rss.xml')#Yahoo
    vocabList,pSF,pNY = localWords(ny,sf)

def testTopWords():
    import feedparser
    ny=feedparser.parse('http://www.nasa.gov/rss/dyn/image_of_the_day.rss')#https://newyork.craigslist.org/search/res?format=rss
    sf=feedparser.parse('http://sports.yahoo.com/nba/teams/hou/rss.xml')#https://sfbay.craigslist.org/search/apa?format=rss
    getTopWords(ny,sf)
#其它RSS 机器学习书籍上的 参考：https://blog.csdn.net/weixin_43744799/article/details/86087542
def getTopWords(ny,sf):
    import operator
    #获取训练数据的条件概率 和精选的词列表
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[]; topSF=[]
    #返回大于某个阈值（注意概率是取对数后的）的词
    for i in range(len(p0V)):
        if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))#获得词以及对应的先验概率（类为0，此词出现的概率）
        if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))
    #输出类0下先验概率超过某个阈值的词
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print(item[0])
    #输出类1下先验概率超过某个阈值的词
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print(item[0])
        
#分类侮辱类和非侮辱类的文本
def test42():
    print('\n*** Load DataSet ***')
    listOPosts,listClasses = loadDataSet()
    print('List of posts:\n', listOPosts)
    print('List of Classes:\n', listClasses)

    print('\n*** Create Vocab List ***')
    myVocabList = createVocabList(listOPosts)
    print('Vocab List from posts:\n', myVocabList)

    print('\n*** Vocab show in post Vector Matrix ***')
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(bagOfWords2Vec(myVocabList,postinDoc))
    print('Train Matrix:\n', trainMat)

    print('\n*** Train ***')
    p0V,p1V,pAb = trainNB0(trainMat,listClasses)
    print('p0V:\n',p0V)
    print('p1V:\n',p1V)
    print('pAb:\n',pAb)

#从RSS源获取文档数据 计算分类的错误率    
testRSS()
#显示地域相关的用词、以获取地域倾向
testTopWords()