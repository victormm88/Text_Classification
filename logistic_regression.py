#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''logistic regression for text classafication'''

__author__ = 'Wang Junqi'

from math import exp;
import numpy as np;
import csv;
import pickle;

#cal sigmoid function
def sigmoid(x):
    for i in xrange(x.shape[0]):
        for j in xrange(x[i].size):
            temp_c=x[i,j]
            temp_c=1/(1+exp(-temp_c));
            x[i,j]=temp_c;


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

#梯度下降法，求解权值矢量
def gradient_decent(W,X,Y,a,l2):
    m=X.shape[1];
    ascent_ratio=1.07;
    decent_ratio=0.93;
    while 1:
        pre_Y=np.dot(W,X);
        sigmoid(pre_Y);
        temp_Y=pre_Y-Y;
        temp_decent=np.dot(temp_Y,X.T);
        temp_decent=(temp_decent+l2*W)/m;
        old_cost=cal_cost(W,X,Y,l2)[1];
        temp_W=W-a*temp_decent;
        new_cost=cal_cost(temp_W,X,Y,l2)[1];
        last_cost=old_cost;
        while new_cost>old_cost:
            a*=decent_ratio;
            temp_W=W-a*temp_decent;
            new_cost=cal_cost(temp_W,X,Y,l2)[1];
        while new_cost<old_cost:
            a*=ascent_ratio;
            temp_W=W-a*temp_decent;
            new_cost=cal_cost(temp_W,X,Y,l2)[1];
            if new_cost>last_cost:
                break;    
            last_cost=new_cost;
        a/=ascent_ratio;
        W=W-a*temp_decent;
        new_cost=cal_cost(W,X,Y,l2)[1];
        if old_cost-new_cost<0.000001:
            break;
        print new_cost;
    return W;

def main():
    cla_dir=load_classinfo('information.txt');
    X,Y=load_matrix('vector_data.txt',cla_dir);
    W=np.random.rand(Y.shape[0],X.shape[0]);
    W=gradient_decent(W,X,Y,0.1,0);
    f_w=open('__W','wb');
    W.dump(f_w);
    f_w.close();

np.set_printoptions(threshold='nan'); 
main();
