#!/usr/bin/env python
# -*- coding: utf-8 -*-

' TF/IDF '

__author__ = 'Wang Junqi'

import os;
import re;
from word import Word;
import math;
import numpy as np;
word_list=[];
classification={};
f=open('information.txt','r');
temp_lines=f.readlines();
f.close();
for x in temp_lines:
    x=x.rstrip('\n').rstrip('\r');
    temp_list=x.split(' ');
    classification[temp_list[0]]=int(temp_list[1]);
num_essay=sum(classification.values());#总文章的数量
f=open('Words_Info.txt','r');
temp_lines=f.readlines();
f.close();
for line in temp_lines:
    line=line.rstrip('\n').rstrip('\r');
    temp_list=line.split(' ');
    temp_word=Word(temp_list[0]);
    temp_word.id=int(temp_list[1]);
    temp_word.count=int(temp_list[2]);
    temp_word.count_essay=int(temp_list[3]);
    temp_index=4;
    for x in classification.keys():
        setattr(temp_word,x,int(temp_list[temp_index]));
        temp_index+=1;
    setattr(temp_word,'idf',math.log(num_essay/temp_word.count_essay));
    word_list.append(temp_word);
temp_list=[(z.name,z.id) for z in word_list];
essay_list=[];
word_dir={x:int(y) for x,y in temp_list};
train_f=open('vector_data.txt','w');
for dir in os.listdir('training'):
    for essay in os.listdir('training/'+dir):
        f=open('training/'+dir+'/'+essay,'r');
        whole_str=f.read();
        f.close();
        whole_list=re.split(r'[^a-z^A-z]',whole_str);
        whole_list=[x.lower() for x in whole_list if x.isalpha()];
        num_words=len(whole_list);
        temp_vector=np.zeros((len(word_list),1));
        for x in whole_list:
            if word_dir.has_key(x):
                temp_vector[word_dir[x],0]+=1;
        temp_vector=1.0*temp_vector/num_words;
        temp_idf=np.array([[x.idf for x in word_list]]).T;
        temp_vector=temp_vector*temp_idf;
        temp_vector=[x[0] for x in temp_vector.tolist()];
        temp_dict={};
        for i in range(len(word_list)):
            if temp_vector[i]!=0:
                temp_dict.setdefault(i,temp_vector[i]);
        for word in word_list:
            if temp_dict.has_key(word.id):
                train_f.write(str(temp_dict[word.id])+',');
            else:
                train_f.write(str(0)+',');
        train_f.write(dir);
        train_f.write('\n');
train_f.close();
# train_f.write(','.join(map(str,temp_vector))+'\n');
# print word_dir;
