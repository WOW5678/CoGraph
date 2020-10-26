# -*- coding:utf-8 -*-
"""
@Time: 2019/06/16 17:23
@Author: Shanshan Wang
@Version: Python 3.7
@Function: 完成一些对数据集的规整工作，比如ICD code中加入分隔符
"""
from collections import defaultdict
import csv
import math
import numpy as np
import sys


mimic_iii_dir_1=r'F:\数据资源\MIMIC\MIMIC-1'
mimic_iii_dir_2=r'F:\数据资源\MIMIC\MIMIC-2'
processed_data_dir = r'F:\MYPAPERS\GraphMatch\code\data'

def reformat(code,is_diag):
    '''
    Put a period in the right place because the MIMIC-3 data files exclude them.
        Generally, procedure codes have dots after the first two digits,
        while diagnosis codes have dots after the first three digits.
    :param code:
    :param is_diag:
    :return:
    '''
    code=''.join(code.split('.'))
    if is_diag:
        if code.startswith('E'):
            if len(code)>4:
                code=code[:4]+'.'+code[4:]
        else:
            if len(code)>3:
                code=code[:3]+'.'+code[3:]
    else:
        code=code[:2]+'.'+code[2:]
    return code

def load_code_descriptions(version='mimic3'):
    #load descriptions lookup from the appropriate data files
    desc_dict=defaultdict(str)
    if version=='mimic2':
        pass
    else:
        with open("%s/D_ICD_DIAGNOSES.csv/D_ICD_DIAGNOSES.csv" % (mimic_iii_dir_1), 'r') as descfile:
            r = csv.reader(descfile)
            # header
            next(r)
            for row in r:
                code = row[1]
                desc = row[-1]
                desc_dict[reformat(code, True)] = desc
        with open("%s/D_ICD_PROCEDURES.csv/D_ICD_PROCEDURES.csv" % (mimic_iii_dir_1), 'r') as descfile:
            r = csv.reader(descfile)
            # header
            next(r)
            for row in r:
                code = row[1]
                desc = row[-1]
                if code not in desc_dict.keys():
                    desc_dict[reformat(code, False)] = desc
        with open('%s/ICD9_descriptions.txt' %mimic_iii_dir_2, 'r') as labelfile:
            for i, row in enumerate(labelfile):
                row = row.rstrip().split()
                code = row[0]
                if code not in desc_dict.keys():
                    desc_dict[code] = ' '.join(row[1:])
    return desc_dict
