#-*-coding:utf-8-*-
import os
#from numpy import array
import cPickle as cpk#用于对中间结果的序列化反序列化
import numpy
from nltk.corpus import wordnet
import time
from sklearn import neighbors
from sklearn.naive_bayes import MultinomialNB 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn import metrics   
import math
import datetime as dt
vocabList=[]#词表列表
label2Id={}#从文本类别标签到文本类别ID的转换（文件夹的文件名作为类别标签）

def loadData(path):
    global label2Id
    folderList=os.listdir(path);#print folderList;
    textList=[];labelList=[];labelId=0
    for folder in folderList:
        if folder not in label2Id.keys(): label2Id[folder]=labelId;labelId+=1
        fileList=os.listdir(path+"/"+folder);#print fileList
        for trainFile in fileList:
            fileObj=open(path+"/"+folder+"/"+trainFile,"r")
            try:
                fileText=fileObj.read()
                textList.append(fileText)
                labelList.append(label2Id[folder])
            finally:
                fileObj.close()
    print labelList[10:20]
    print label2Id
    print len(labelList)
    print len(textList)
    return textList,labelList#textList文本列表，labelList类别标签列表

def removeSpecialSymbol(text,symbol):#主要用于移除逗号句号的同时避免移除小数点等，symbol:"."","
    index=0;
    while index!=-1:
        index=text.find(symbol,index);#print index
        if index==-1: break;
        if index==0:
            text=text[1:]
        elif index==(len(text)-1):
            text=text[:index]
            break
        else:
            if index<len(text)-1 and not(text[index-1].isdigit() and text[index+1].isdigit()):
                text=text[:index]+text[index+1:]#+" "
                #index+=1
            else:
                index+=1#index=text.find(symbol,index+1)
    return text

def textPreprocess(text):#预处理text，输出结为词列表
    
    text=text.lower().strip()
    text=text.replace("-"," ").replace("\""," ").replace("'s"," ").replace("'","").replace("("," ").replace(")"," ").replace("<"," ").replace(">"," ").strip()#.replace(","," ")
    text=removeSpecialSymbol(text,".")
    text=removeSpecialSymbol(text,",")
    #print text
    wordList=text.split();#
    #print wordList

    with open("stopwords.txt","r") as fileObj:
        lines=fileObj.readlines()
        stopwords=[word.strip() for word in lines]
        stopwords=set(stopwords)
        wordList=[word for word in wordList if word not in stopwords]
        wordList=[word for word in wordList if len(word)>1]

    wordIndex=0
    while wordIndex<len(wordList):
        if len(wordList[wordIndex])>3 and (wordList[wordIndex][len(wordList[wordIndex])-3:]=="ing" or \
                                           wordList[wordIndex][len(wordList[wordIndex])-2:]=="ed"):
            wordPrototype=wordnet.morphy(wordList[wordIndex],pos="v")
        else: wordPrototype=wordnet.morphy(wordList[wordIndex])#还原词干
        if wordPrototype!=None and wordList[wordIndex]!=wordPrototype:
            wordList[wordIndex]=wordPrototype   
        wordIndex+=1    
    return wordList

def getVocab(textList):#获得词表
    global vocabList;processCNT=0
    for text in textList:
        vocabList.extend(textPreprocess(text));#i+=1#print text;
        processCNT=processCNT+1
        print "getVocab process:"+str(processCNT)+"\n"
    vocabList=list(set(vocabList))
    vocabList.sort()
    #return vocabList

def getPreprocessedTextList(textList):#获得用词列表表示的文档，以及词表
    processedTextList=[];global vocabList
    temp=[]
    for text in textList:
        temp=textPreprocess(text)
        processedTextList.append(temp)
        vocabList.extend(temp)
    vocabList=list(set(vocabList))
    vocabList.sort()
    return processedTextList

def preprocessText(textList):
    processedTextList=[]
    for text in textList:
        processedTextList.append(textPreprocess(text))
    return processedTextList

def text2vec(textList,vocabList):#（词列表表示的文档，词表）
    docVecList=[];text2vecCnt=0
    #print vocabList
    #print vocabList
    for text in textList:
        docVec=numpy.array([0]*len(vocabList))
        for word in text:
            if word in vocabList: docVec[vocabList.index(word)]+=1
        #docVec=docVec/float(len(wordList))##########
        docVec=[word/float(len(text)) for word in docVec]
        docVecList.append(docVec)
        text2vecCnt=text2vecCnt+1
        print "processed:"+str(text2vecCnt)+"\n"
    #print vocabList[30000:31000]
    #print len(vocabList)
    #print docVecList[0]
    #docVecList=array(docVecList)
    return docVecList#文档向量列表

def chiSquareTest(docVecList,labelList):#docVecList:文档的词频向量，labelList:文档标签列表
    featureList=[]
    featureCntEachClass=100#每个类别选取的特征数
    vocabCnt=len(vocabList)#词表词量
    labelTypeCnt=len(set(labelList))#文档类别数
    chiSquareVal=numpy.zeros((vocabCnt,labelTypeCnt))#文档的卡方值矩阵
    #n下标1为类别，下标2为词，下标一为1表示出现某词，下标二为1表示在某类中
    #n11(i,j):词i出现在类别j中的文档数
    #n01(i,j):类别j的文档不出现词i的文档数
    #n10(i,j):非类别j的文档出现词i的文档数
    #n00(i,j):非类别j的文档出不现词i的文档数
    n11=numpy.zeros((vocabCnt,labelTypeCnt))
    n01=numpy.zeros((vocabCnt,labelTypeCnt))
    n10=numpy.zeros((vocabCnt,labelTypeCnt))
    n00=numpy.zeros((vocabCnt,labelTypeCnt))
    #n10=array([0]*len(vocabList))
    #n00=array([0]*len(vocabList))
    for i in xrange(vocabCnt):#第i个词
        for j in xrange(len(docVecList)):#第j个文档
            if docVecList[j][i]!=0:
                n11[i][labelList[j]]+=1#为第i个词在第j个文档所处的类别的出现+1
            else: n01[i][labelList[j]]+=1
    for k in xrange(vocabCnt):
        for l in xrange(labelTypeCnt):
            for m in xrange(labelTypeCnt): 
                if m!=l: 
                   n10[k][l]+=n11[k][m]
    for x in xrange(vocabCnt):
        for y in xrange(labelTypeCnt):
            for z in xrange(labelTypeCnt):
                if z!=y: 
                   n00[x][y]+=n01[x][z]
    for p in xrange(vocabCnt):
        for q in xrange(labelTypeCnt):
            chiSquareVal[p][q]=(n11[p][q]+n10[p][q]+n01[p][q]+n00[p][q])*(n11[p][q]*n00[p][q]-n10[p][q]*n01[p][q])**2 \
                               /(n11[p][q]+n01[p][q])*(n11[p][q]+n10[p][q])*(n10[p][q]+n00[p][q])*(n01[p][q]+n00[p][q])
    featureRank=numpy.argsort(-chiSquareVal,axis=0)#对每矩阵一列进行降序排序，返回的矩阵元素值为该列元素编号
    for r in xrange(labelTypeCnt):
        for s in  range(featureCntEachClass):
            featureList.append(featureRank[s][r])
    featureList=list(set(featureList))
    featureList.sort()  
    return featureList#特征列表，仅特征序号

def featureSelect(docVecList,featureList):
    docVecListNew=[]
    #featureList=chiSquareTest(docVecList,labelList)
    for i in xrange(len(docVecList)):
        docVecNew=[]
        for j in xrange(len(featureList)):
            docVecNew.append(docVecList[i][featureList[j]])
        docVecListNew.append(docVecNew)
    return docVecListNew#特征选择后的词频列表
    
def calc_f1(actual,pred):  
    m_precision = metrics.precision_score(actual,pred);  
    m_recall = metrics.recall_score(actual,pred);  
    print 'predict info:'  
    print 'precision:{0:.3f}'.format(m_precision)  
    print 'recall:{0:0.3f}'.format(m_recall);  
    print 'f1-score:{0:.3f}'.format(metrics.f1_score(actual,pred));  

def main1():
    startTime=time.clock()
    textListTrain,labelListTrain=loadData("top10_training")
    textListTest,labelListTest=loadData("top10_test")
    #textListAll=[]
    #textListAll.extend(textListTrain);textListAll.extend(textListTest)
    numTrain=len(textListTrain);numTest=len(textListTest)
    getVocab(textListTrain)
    docVecList=text2vec(textListTrain,vocabList);docVecListTest=text2vec(textListTest,vocabList)
    print "docNum:"+str(len(docVecList))
    print "vocabCnt:"+str(len(vocabList))
    print "docVecList"+str(numpy.shape(docVecList))
    print "docVecListTest"+str(numpy.shape(docVecListTest))
    featureList=chiSquareTest(docVecList,labelListTrain)


    with open(r'serialization/featureList.txt','w') as f:
        cpk.dump(featureList,f)
    with open('serialization/docVecList.txt','w') as f1:
        cpk.dump(docVecList,f1)
    with open('serialization/docVecListTest.txt','w') as f2:
        cpk.dump(docVecListTest,f2)
    with open('serialization/labelListTrain.txt','w') as f3:
        cpk.dump(labelListTrain,f3)
    with open('serialization/labelListTest.txt','w') as f4:
        cpk.dump(labelListTest,f4)

    timeUsed=time.clock()-startTime
    print " timeused："+str(timeUsed)

def main2():
    startTime=time.clock()
    with open(r'serialization/featureList.txt') as f:
        featureList=cpk.load(f)
    with open('serialization/docVecList.txt') as f1:
        docVecList=cpk.load(f1)
    with open('serialization/docVecListTest.txt') as f2:
        docVecListTest=cpk.load(f2)
    with open('serialization/labelListTrain.txt') as f3:
        labelListTrain=cpk.load(f3)
    with open('serialization/labelListTest.txt') as f4:
        labelListTest=cpk.load(f4)
    numTest=len(docVecListTest)
    docVecList=featureSelect(docVecList,featureList)
    docVecListTest=featureSelect(docVecListTest,featureList)
    time_knnstart=time.clock()
    print "preprocess time used:"+str(time_knnstart-startTime)
    print "labelListTest_count:"+str(len(labelListTrain))+"labelListTest_count:"+str(len(labelListTest))
    print "vectorsize:"+str(len(docVecList[0]))
    knn=neighbors.KNeighborsClassifier()
    knn.fit(numpy.array(docVecList),numpy.array(labelListTrain))
    '''numRight=0;index=0
    for vec in docVecListTest:
        if knn.predict(numpy.array(vec))[0]==labelListTest[index]: numRight+=1
        index+=1
    accuracy=numRight/float(numTest)
    time_knn=time.clock()
    print "knn accuracy："+str(accuracy)+" timeused："+str(time_knn-startTime)'''
    pred0=knn.predict(numpy.array(docVecListTest))
    calc_f1(numpy.array(labelListTest),pred0)
    time_knn=time.clock()
    print "knn time used:"+str(time_knn-time_knnstart)

    nb=MultinomialNB(alpha=0.01)
    nb.fit(numpy.array(docVecList),numpy.array(labelListTrain))
    pred=nb.predict(numpy.array(docVecListTest))
    calc_f1(numpy.array(labelListTest),pred)
    time_nb=time.clock()
    print "nb time used:"+str(time_nb-time_knn)
    
    svm=SVC(kernel='linear')
    svm.fit(numpy.array(docVecList),numpy.array(labelListTrain))
    pred1=svm.predict(numpy.array(docVecListTest))
    calc_f1(numpy.array(labelListTest),pred1)
    time_svm=time.clock()
    print "svm time used:"+str(time_svm-time_nb)
    
    lr=LogisticRegression().fit(numpy.array(docVecList),numpy.array(labelListTrain))
    pred2=lr.predict(numpy.array(docVecListTest))
    calc_f1(numpy.array(labelListTest),pred2)
    time_lr=time.clock()
    print "lr time used:"+str(time_lr-time_svm)






def tf_idf(textList,vocabList):#（词列表表示的文档，词表）
    docVecList=[]
    textCnt=len(textList);wordCnt=len(vocabList)
    docMat=numpy.zeros((textCnt,wordCnt))
    vocabSet=set(vocabList)
    tfCnt=0
    for i in xrange(textCnt):
        #docDic={word:0 for word in vocabList}##list转换成dict
        for word in textList[i]:
            if word in vocabSet: docMat[i,vocabList.index(word)]+=1
        #docVec=docVec/float(len(wordList))##########
        numpy.divide(docMat[i,:],len(textList[i]),docMat[i,:])
        tfCnt+=1; print "tf processed:"+str(tfCnt)+"\n"

    idfCnt=0;docFreq=0
    for j in xrange(wordCnt):
        docFreq=0
        for i in xrange(textCnt):
            if docMat[i][j]>0:
                docFreq+=1
        if(docFreq!=0):#通过训练集得到的词表，词表中有些词在测试集未出现
            idf=math.log(float(textCnt)/docFreq,2)
            #numpy.divide(arr1,arr2,arr)   arr=arr1/arr2
            numpy.multiply(docMat[:,j],idf,docMat[:,j])
        idfCnt+=1
        print "idf processed:"+str(idfCnt)+"\n"
    return docMat
    
    
    
    

def featureSelect_tfidf(docMat):
    featureCntEachText=1
    featureList=[]
    textCnt=docMat.shape[0]
    featureRank=numpy.argsort(-docMat,axis=1)#对每矩阵一列进行降序排序，返回的矩阵元素值为该列元素编号
    for r in xrange(textCnt):
        for s in  xrange(featureCntEachText):
            featureList.append(featureRank[r][s])
    featureList=list(set(featureList))
    featureList.sort()
    return featureList#新特征列表，仅仅是在原词表中的编号

def reducedMatrix(docMat,featureList):
    featureCnt=len(featureList);textCnt=docMat.shape[0]
    docMatNew=numpy.zeros((textCnt,featureCnt))
    for i in xrange(textCnt):
        for j in xrange(featureCnt):
            docMatNew[i,j]=docMat[i,featureList[j]]
    return docMatNew#文档向量列表
    

def tfidf(docVecArr):#docVecArr:numpy.array[][]   row:text  column:word element:word frequency in doc float type
    cntDoc=docVecArr.shape[0]
    cntWord=docVecArr.shape[1];docProccessed=0
    for j in range(cntWord):
        docFreq=0;docProccessed=docProccessed+1;print 'tfidf-word:'+str(docProccessed)+"\n"
        for i in range(cntDoc):
            if docVecArr[i][j]>0:
                docFreq+=1
        idf=math.log(float(cntDoc)/docFreq,2)
        numpy.multiply(docVecArr[:,j],idf,docVecArr[:,j])#numpy.divide(arr1,arr2,arr)   arr=arr1/arr2
    print "docVecArr[0]:"+str(docVecArr[0])
    return docVecArr



def TEST():
    textListTrain,labelListTrain=loadData("top10_training")
    textListTest,labelListTest=loadData("top10_test")
    textTest=preprocessText(textListTest)
    getVocab(textListTrain)
    docVecListTest=text2vec(textTest[2275:2278],vocabList)
    print textTest[2276]
    print docVecListTest[1]
    print "\n" 
    print textTest[2277]  
    print docVecListTest[2]

def main3():
    time1=dt.datetime.now()
    
    textListTrain,labelListTrain=loadData("top10_training")
    textListTest,labelListTest=loadData("top10_test")
    numTrain=len(textListTrain);numTest=len(textListTest)
    getVocab(textListTrain)

    global vocabList
    textTrain=preprocessText(textListTrain)
    docMat=tf_idf(textTrain,vocabList)
    feaList=featureSelect_tfidf(docMat)#feaList中仅为新特征在原词表中的编号
    docVecArr=reducedMatrix(docMat,feaList)
    with open('serialization/docVecArr1.txt','wb') as f1:
        cpk.dump(docVecArr,f1)


    textTest=preprocessText(textListTest)
    docMatTest=tf_idf(textTest,vocabList)
    docVecArrTest=reducedMatrix(docMatTest,feaList)
    with open('serialization/docVecArrTest1.txt','wb') as f2:
        cpk.dump(docVecArrTest,f2)

    

    
    
    
    time2=dt.datetime.now()
    print "preprocess time used(s):"+str(time2-time1)   
 

    print "train:"+str(len(docVecArr))+"feature:"+str(len(docVecArr[0]))
    print "test:"+str(len(docVecArrTest))+"feature:"+str(len(docVecArrTest[0]))
    print "feature Cnt:"+str(docVecArr.shape[1])
   
    with open('serialization/labelListTrain.txt','w') as f3:
        cpk.dump(labelListTrain,f3)
    with open('serialization/labelListTest.txt','w') as f4:
        cpk.dump(labelListTest,f4)
 


def main4():
    startTime=dt.datetime.now()
    
    with open('serialization/docVecArr1.txt','rb') as f1:
        docVecArr=cpk.load(f1)
    with open('serialization/docVecArrTest1.txt','rb') as f2:
        docVecArrTest=cpk.load(f2)
    with open('serialization/labelListTrain.txt') as f3:
        labelListTrain=cpk.load(f3)
    with open('serialization/labelListTest.txt') as f4:
        labelListTest=cpk.load(f4)
    print  "doc Count:"+str(docVecArr.shape[0])
    print  "feature size:"+str(docVecArr.shape[1])
    

    time_knnstart=dt.datetime.now()
    print "preprocess time used:"+str(time_knnstart-startTime)
    print "labelListTest_count:"+str(len(labelListTrain))+"labelListTest_count:"+str(len(labelListTest))
    labelArrTrain=numpy.array(labelListTrain)
    labelArrTest=numpy.array(labelListTest)

    knn=neighbors.KNeighborsClassifier()
    knn.fit(docVecArr,labelArrTrain)
    pred0=knn.predict(docVecArrTest)
    calc_f1(labelArrTest,pred0)
    time_knn=dt.datetime.now()
    print "knn time used:"+str(time_knn-time_knnstart)

    nb=MultinomialNB(alpha=0.01)
    nb.fit(docVecArr,labelArrTrain)
    pred=nb.predict(docVecArrTest)
    calc_f1(labelArrTest,pred)
    time_nb=dt.datetime.now()
    print "nb time used:"+str(time_nb-time_knn)
    
    svm=SVC(kernel='linear')
    svm.fit(docVecArr,labelArrTrain)
    pred1=svm.predict(docVecArrTest)
    calc_f1(labelArrTest,pred1)
    time_svm=dt.datetime.now()
    print "svm time used:"+str(time_svm-time_nb)
    
    lr=LogisticRegression().fit(docVecArr,labelArrTrain)
    pred2=lr.predict(docVecArrTest)
    calc_f1(labelArrTest,pred2)
    time_lr=dt.datetime.now()
    print "lr time used:"+str(time_lr-time_svm)

    


