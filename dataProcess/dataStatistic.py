# -*- coding:utf-8 -*-
"""
@Time: 2019/07/26 19:16
@Author: Shanshan Wang
@Version: Python 3.7
@Function: 数据统计
"""
import csv
import pandas as pd
def get_diagnosis_procedure(filename):
    with open(filename) as f:
        reader=csv.reader(f)
        next(reader)
        data=[row for row in reader]
        icdCount=[]
        icdNum=[]
        entityCount=[]
        entityNum=[]
        for row in data:
            icdL=row[3].replace(' ','').split(';')
            icdCount.extend(icdL)
            icdNum.append(len(icdL))

            entityL=row[4].replace(' ','').split('\t')
            entityCount.extend(entityL)
            entityNum.append(len(entityL))

        icdCount=set(icdCount)
        print(icdCount)
        print(len(icdCount))
        print(sum(icdNum)/len(icdNum))

        entityCount=set(entityCount)
        print(len(entityCount))
        print(sum(entityNum)/len(entityNum))

def wikipediaEntity(filename):
    with open(filename) as f:
        reader=csv.reader(f)
        next(reader)
        data=[row for row in reader]
        entityCount=[]
        entityNum=[]
        for row in data:
            entityL=row[2].split('\t')
            entityCount.extend(entityL)
            entityNum.append(len(entityL))

        entityCount=set(entityCount)
        print(len(entityCount))
        print(sum(entityNum)/len(entityNum))



if __name__ == '__main__':
    #get_diagnosis_procedure('../data/note_labeled_entity.csv')
    wikipediaEntity('../data/ICD_descript_entity.csv')