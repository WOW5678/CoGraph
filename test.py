# -*- coding:utf-8 -*-
"""
@Time: 2020/09/29 19:39
@Author: Shanshan Wang
@Version: Python 3.7
@Function:
"""
import torch
import random

a=[[1,2,3],[1,2,3]]
b=[]
b.extend(a)
print(b)
c=list(zip(a,b))
print('c:',c)
x=random.sample(c,2)
print(x)