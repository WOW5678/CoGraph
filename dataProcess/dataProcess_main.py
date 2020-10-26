# -*- coding:utf-8 -*-
"""
@Time: 2019/06/17 21:37
@Author: Shanshan Wang
@Version: Python 3.7
@Function: 数据预处理的主文件
"""
# --*- coding:utf-8 -*-
import sys
sys.path.append('code/dataProcess')
#import log_reg
#import extract_wvs
from dataProcess.get_discharge_summaries import write_discharge_summaries
from dataProcess import concat_and_split
from dataProcess import build_vocab
from dataProcess import vocab_index_descriptions
from dataProcess import word_embeddings
from dataProcess import datasets

import numpy as np
import pandas as pd

from collections import Counter,defaultdict
import csv
import math
import operator

processed_data_dir = r'F:\MYPAPERS\GraphMatch\code\data'
mimic_iii_dir_1=r'F:\数据资源\MIMIC\MIMIC-1'
mimic_iii_dir_2=r'F:\数据资源\MIMIC\MIMIC-2'
# 定义一些变量
Y = 'full'  # 表示使用全部的ICD codes
note_files = r'F:\数据资源\MIMIC\MIMIC-2\MIMIC-2\NOTEEVENTS.csv\NOTEEVENTS.csv'
vocab_size = 'full'  # 表示不限定vocab的个数，即保留所有的vocab
vocab_min = 3  # 出现次数小于3次的单词会被抛弃

if __name__ == '__main__':

    #Combine diagnosis and procedure codes and reformat them
    # EHR中的Code 没有使用.分割，为了防止诊断code和procedure code产生冲突，我们需要为每个code添加上分隔符
    dfproc=pd.read_csv(open('%s\PROCEDURES_ICD.csv\PROCEDURES_ICD.csv'%mimic_iii_dir_2,'r'),header=None,names=[ 'ROW_ID','SUBJECT_ID','HADM_ID','SEQ_NUM','ICD9_CODE'])
    dfdiag=pd.read_csv(open('%s\DIAGNOSES_ICD.csv\DIAGNOSES_ICD.csv'%mimic_iii_dir_1,'r'),header=None,names=[ 'ROW_ID','SUBJECT_ID','HADM_ID','SEQ_NUM','ICD9_CODE'])
    print(dfproc.head(5)) #columns:["ROW_ID","SUBJECT_ID","HADM_ID","SEQ_NUM","ICD9_CODE"]
    print(dfdiag.head(5)) #columns:["ROW_ID","SUBJECT_ID","HADM_ID","SEQ_NUM","ICD9_CODE"]
    print(dfdiag["ICD9_CODE"])
    dfdiag['absolute_code']=dfdiag.apply(lambda row:str(datasets.reformat(str(row[4]),True)),axis=1)
    dfproc['absolute_code']=dfproc.apply(lambda row:str(datasets.reformat(str(row[4]),False)),axis=1)
    dfcodes = pd.concat([dfdiag, dfproc])  # 在纵轴上拼接两个df,即行数增加
    print(dfcodes.head()) #ROW_ID  SUBJECT_ID  HADM_ID  SEQ_NUM  ICD9_CODE    ICD.9_CODE
    # #len(dfcodes) 891142
    # dfcodes.to_csv('%s\ALL_CODES.csv'%processed_data_dir, index=False,
    #               columns=['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'SEQ_NUM', 'absolute_code'],
    #               header=['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'SEQ_NUM', 'ICD9_CODE'])
    #
    # Tokenize and preprocess raw text
    # Select only discharge summaries and their addenda
    # remove punctuation and numeric-only tokens, removing 500 but keeping 250mg
    # lowercase all token
    disch_full_file = write_discharge_summaries(out_file='%s\disch_full.csv'%processed_data_dir)
    dfdisch_full = pd.read_csv('%s\disch_full.csv'%processed_data_dir,
                     #names=['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'TEXT']
                                )
    # Tokens and types
    # types = set()
    # num_tok = 0
    # for row in dfdisch_full.itertuples():
    #     print(row[4])
    #     for w in row[4].split():
    #         types.add(w)
    #         num_tok += 1
    # print('Number types:', len(types)) #150855
    # print('Number tokens:', num_tok)   #79801388

    ##Let's sort by SUBJECT_ID and HADM_ID to make a correspondence with the MIMIC-3 label file
    #将dfdisch_full的subject_id 转换成id ,否则的话排序不准确
    #print(dfdisch_full['SUBJECT_ID']) # int64
    # dfdisch_full=dfdisch_full.apply(lambda row:int(row[0]))
    # dfdisch_full=dfdisch_full.apply(lambda row:int(row[1]))
    dfdisch_full = dfdisch_full.sort_values(['SUBJECT_ID', 'HADM_ID'])
    ##Sort the label file by the same
    dfcodes=dfcodes.sort_values(['SUBJECT_ID', 'HADM_ID'])
    print(len(dfdisch_full['HADM_ID'].unique()),len(dfcodes['HADM_ID'].unique())) #52726 77402
    #这说明有些codes没有对应的discharge文档

    #Consolidate labels with set of discharge summaries
    #过滤掉那些没有对应文档的codes
    #这部分代码只需要执行一次
    hadm_ids = set(dfdisch_full['HADM_ID'])
    print(hadm_ids)
    with open('%s/ALL_CODES.csv' % processed_data_dir, 'r') as lf:
        with open('%s/ALL_CODES_filtered.csv' % processed_data_dir, 'w',newline='') as of:
            w = csv.writer(of)
            w.writerow(['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE', 'ADMITTIME', 'DISCHTIME'])
            r = csv.reader(lf)
            # header
            next(r)
            for row in r:
                hadm_id = int(row[2])
                #print(type(hadm_id))
                if hadm_id in hadm_ids:
                    w.writerow(row[1:3] + [row[-1], '', ''])

    dfcodes_filtered=pd.read_csv(open('%s/ALL_CODES_filtered.csv' % processed_data_dir),index_col=None)
    print(len(dfcodes_filtered['HADM_ID'].unique())) #52726
    ##we still need to sort it by HADM_ID
    dfcodes_filtered = dfcodes_filtered.sort_values(['SUBJECT_ID', 'HADM_ID'])
    dfcodes_filtered.to_csv('%s/ALL_CODES_filtered.csv' % processed_data_dir, index=False)

    #Append labels to notes in a single file

    # Now let's append each instance with all of its codes
    # this is pretty non-trivial so let's use this script I wrote, which requires the notes to be written to file
    sorted_file = '%s/disch_full.csv' % processed_data_dir
    dfdisch_full.to_csv(sorted_file, index=False)

    labeled=concat_and_split.concat_data('%s/ALL_CODES_filtered.csv' % processed_data_dir, sorted_file)
    print(labeled) #note_labeled.csv

    dflabeled=pd.read_csv(open(labeled))
    '''
    #tokens and types
    types=set()
    num_tok=0
    for row in dflabeled.itertuples():
        for w in row[3].split():
            types.add(w)
            num_tok+=1
    print("num types", len(types), "num tokens", num_tok) #num types 144394 num tokens 71482894
    print(len(dflabeled['HAMD_ID'].unique()))
    '''

    ############################################################################
    ############################################################################
    #以下的部分要根据不同的试验进行调整
    #Create train/dev/test splits
    train,dev,test=concat_and_split.split_data(labeled,base_name='%s\disch'%processed_data_dir)

    # Create vocabluary from training data
    # vocab_min=3
    # vname='%s/vocab.csv'%processed_data_dir
    #build_vocab.build_vocab(vocab_min,train,vname)

    # Sort each data split by length for batching
    # for split in ['train', 'dev', 'test']:
    #     filename = '%s/disch_%s_split.csv' % (processed_data_dir,split)
    #     df = pd.read_csv(filename)
    #     df['length'] = df.apply(lambda row: len(str(row['TEXT']).split()), axis=1)
    #     df = df.sort_values(['length'])
    #     df.to_csv('%s/%s_full.csv' % (processed_data_dir, split), index=False)

    # Pre-train word embeddings
    #Let's train word embeddings on all words
    #w2v_file=word_embeddings.word_embeddings('full','%s/disch_full.csv' % processed_data_dir, 100, 0, 5)

    #Write pre-trained word embeddings with new vocab
    #word_embeddings.gensim_to_emebddings('%s/processed_full.w2v' % processed_data_dir, '%s/vocab.csv' % processed_data_dir, Y)

    #Pre-process code descriptions using the vocab
    # vocab_index_descriptions.vocab_index_descriptions('%s/vocab.csv' % processed_data_dir,
    #                                                   '%s/description_vectors.vocab' % processed_data_dir)