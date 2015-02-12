#!/usr/bin/env python
# -*- coding: utf-8 -*-

' Classifier '

__author__ = 'Wang Junqi'

import math;
import os;
import numpy as np;
from word import Word;
import re;
import KNN;
import csv;


#load classafication infomation
def load_classinfo(file_name):
    temp_dir={};
    f=open(file_name,'r');
    lines=f.readlines();
    for l in xrange(len((lines))):
        line=lines[l].split(' ');
        temp_dir[line[0]]=l;
    return temp_dir;

#creat matrix X Y as training data,X(d+1,n) Y(c,n)
def load_matrix(file_name,cla_dir):
    X=[];
    Y=[];
    f=open(file_name,'rb');
    csv_reader=csv.reader(f);
    for line in csv_reader:
        cla=line.pop();
        line.insert(0,1);
        X.append(line);
        temp_y=[0]*len(cla_dir);
        temp_y[cla_dir[cla]]=1;
        Y.append(temp_y);
    X=np.array(X).transpose().astype(np.float);
    Y=np.array(Y).transpose().astype(np.float);
    return X,Y;

#cal cost with W(c,d+1) X(d+1,n) Y(c,n)
def cal_cost(W,X,Y,l2):
    m=X.shape[1];
    pre_Y=np.dot(W,X);
    sigmoid(pre_Y);
    minY=1-Y;
    pre_minY=1-pre_Y;
    pre_Y=np.log(pre_Y);
    pre_minY=np.log(pre_minY);
    cost_matrix=Y*pre_Y+minY*pre_minY;
    cost=-np.sum(cost_matrix);
    l2_cost=cost+0.5*l2*(np.sum(W**2));
    return (cost/m,l2_cost/m);

def sigmoid(x):
    for i in xrange(x.shape[0]):
        for j in xrange(x[i].size):
            temp_c=x[i,j]
            temp_c=1/(1+math.exp(-temp_c));
            x[i,j]=temp_c;
    return x;

# 载入类别信息
classification={};
cla_list=[];
f=open('information.txt','r');
temp_lines=f.readlines();
f.close();
for x in temp_lines:
    x=x.rstrip('\n');
    temp_list=x.split(' ');
    cla_list.append(temp_list[0]);
    classification[temp_list[0]]=int(temp_list[1]);
num_essay=sum(classification.values());#总文章的数量
# 载入训练样本集
f=open('training_data.txt','r');
temp_lines=f.readlines();
f.close();
essay_list=[];
for line in temp_lines:
    temp_essay_dict={};
    line=line.rstrip('\n');
    word_list=line.split(',');
    for word in word_list:
        if word.find(':')!=-1:
            temp_list=word.split(':');
            temp_essay_dict[int(temp_list[0])]=float(temp_list[1]);
        else:
            temp_essay_dict['class']=word;
    essay_list.append(temp_essay_dict);
# print essay_list;
# 载入词表
f=open('Words_Info.txt','r');
temp_lines=f.readlines();
f.close();
word_list=[];
for line in temp_lines:
    line=line.rstrip('\n');
    temp_list=line.split(' ');
    temp_word=Word(temp_list[0]);
    temp_word.id=int(temp_list[1]);
    temp_word.count=int(temp_list[2]);
    temp_word.count_essay=int(temp_list[3]);
    setattr(temp_word,'idf',math.log(num_essay/temp_word.count_essay));
    word_list.append(temp_word);
temp_list=[(z.name,z.id) for z in word_list];
word_dir={x:int(y) for x,y in temp_list};
# 分类器
# print word_list;
all_essay=0;
right_essay=0;
f_w=open('__W','rb');
W=np.load(f_w);
f_w.close();
print W;
cla_dir=load_classinfo('information.txt');
X,Y=load_matrix('vector_data.txt',cla_dir);
print cal_cost(W,X,Y,0)[0];
np.set_printoptions(threshold='nan'); 
f_result=open('result.txt','w');
for dir in os.listdir('test'):
    for essay in os.listdir('test/'+dir):
        all_essay+=1;
        essay_dict={};
        f=open('test/'+dir+'/'+essay,'r');
        whole_str=f.read();
        f.close();
        whole_list=re.split(r'[^a-z^A-z]',whole_str);
        whole_list=[x.lower() for x in whole_list if x.isalpha()];
        num_words=len(whole_list);
        for word in whole_list:
            if word_dir.has_key(word):
                if essay_dict.has_key(word_dir[word]):
                    essay_dict[word_dir[word]]+=1;
                else:
                    essay_dict[word_dir[word]]=1;
        for x,y in essay_dict.iteritems():
            temp_y=1.0*y/num_words*word_list[x].idf;
            essay_dict[x]=temp_y;
        
        # f_result.write();
        essay_list=[];
        for word in word_list:
            if essay_dict.has_key(word.id):
                essay_list.append(essay_dict[word.id]);
            else:
                essay_list.append(0);
        essay_list.insert(0,1);
        essay_list=np.array(essay_list).reshape(len(essay_list),1);
        pre_y=np.dot(W,essay_list);
        sigmoid(pre_y);
        pre_y=pre_y.tolist();
        #print pre_y;
        C=cla_list[pre_y.index(max(pre_y))];
        #C=KNN.KNN_Classification(essay_dict,essay_list,5);
        print C,dir;
        if C==dir:
            right_essay+=1;
        # f_result.write('true:'+dir+'\t'+'pretect:'+C+'\n');
print right_essay;
print all_essay;
f_result.close();
