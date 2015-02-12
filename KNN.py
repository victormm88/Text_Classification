# -*- coding: utf-8 -*-

' KNN '

__author__ = 'Wang Junqi'

import copy;
import math;

#欧氏距离
def Cal_Dis(dict1,dict2):
	dis=0.0;
	for x in dict1.keys():
		if dict2.has_key(x):
			dis+=((dict1[x]-dict2[x])**2)/2
		else:
			dis+=dict1[x]**2;
	for x in dict2.keys():
		if x!='class':
			if dict1.has_key(x):
				dis+=((dict1[x]-dict2[x])**2)/2
			else:
				dis+=dict2[x]**2;
	return dis;

#余弦相似度
def Cal_Cos(dict1,dict2):
	ab_dict1=0.0;
	ab_dict2=0.0;
	for x in dict1.values():
		ab_dict1+=x**2;
	ab_dict1=math.sqrt(ab_dict1);
	for x in dict2.values():
		if not type(x)==str:
			ab_dict2+=x**2;
	ab_dict2=math.sqrt(ab_dict2);
	time_1_2=0.0;
	for k in dict1.keys():
		if dict2.has_key(k):
			time_1_2+=dict1[k]*dict2[k];
	# print time_1_2/(ab_dict1*ab_dict2);
	return time_1_2/(ab_dict1*ab_dict2);
	
def KNN_Classification(text_dict,inner_training_data,k):
	dis_dict={};
	# inner_training_data=copy.deepcopy(training_data);
	for i in range(len(inner_training_data)):
		dis=Cal_Cos(text_dict,inner_training_data[i]);
		dis_dict[i]=dis;
	dis_list=sorted(dis_dict.items(),key=lambda x:x[1],reverse=True)[:k];
	temp_dict={};
	for item in dis_list:
		if temp_dict.has_key(inner_training_data[item[0]]['class']):
			temp_dict[inner_training_data[item[0]]['class']]+=math.sqrt(item[1]);
		else:
			temp_dict[inner_training_data[item[0]]['class']]=math.sqrt(item[1]);
	temp_list=sorted(temp_dict.items(),key=lambda x:x[1],reverse=True);
	# print temp_list;
	return temp_list[0][0];