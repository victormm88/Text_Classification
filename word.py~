# -*- coding: utf-8 -*-

' µ•¥ ¿‡ 2014/11/13'

__author__ = 'Wang Junqi'

import math;
class Word(object):
	
	def __init__(self,name):
		self.name=name;
		self.count_essay=1;
		self.count=1;
	def get_name(self):
		return self.name;
	def calculate_C(self,dict_class,dict_class_num,num_essay):
		entropy=0.0;
		entropy2=0.0;
		for x in dict_class.keys():
			p=1.0*getattr(self,x)/self.count_essay;
			p2=1.0*(dict_class_num[x]-getattr(self,x))/(num_essay-self.count_essay);
			if p:
				entropy+=p*math.log(p,2);
			if p2:
				entropy2+=p2*math.log(p2,2);
		entropy=entropy*self.count_essay/num_essay;
		entropy2=entropy2*(num_essay-self.count_essay)/num_essay;
		return -math.log(1.0/len(dict_class),2)+entropy+entropy2;
		
		

	