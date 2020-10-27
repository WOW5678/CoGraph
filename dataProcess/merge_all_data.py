# -*- coding:utf-8 -*-
"""
@Time: 2019/06/26 15:06
@Author: Shanshan Wang
@Version: Python 3.7
@Function: 将训练集中不同部分得到的实体在跟原来的训练集数据进行合并
"""
import csv
import pandas as pd
import glob

def merge_data():
    icds=[] #所有的icds
    with open('../data/note_labeled.csv','r') as f:
        reader=csv.reader(f)
        next(reader)
        for row in reader:
            icds.extend(row[3].split(';'))

    #icds=list(set(icds)) #保存了所有独一无二的icd 应该是8921个
    codes = set([c for c in icds if c != ''])
    print('len(icds):', len(codes))
    df_icds=pd.DataFrame(icds)
    print(df_icds.head(5))
    df_icds.columns = ['icd']

    #所有的icd的描述
    with open('../data/ICD9_descriptions.txt','r') as f:
        icd_desriptions=[row.split('\t') for row in f.readlines()]
    icd_desriptions_df=pd.DataFrame(icd_desriptions)
    print(icd_desriptions_df.head(5))
    icd_desriptions_df.columns=['icd','description']
    # 去重
    icd_desriptions_df.drop_duplicates(['icd'],inplace=True)

    #对两者进行合并
    result1=pd.merge(df_icds,icd_desriptions_df,how='left',on=['icd'])
    print(result1.head(5))
    #每个ICD对应的实体
    icd_entity_df=pd.read_csv('../data/ICD_descript_entity.csv')
    icd_entity_df.columns=['icd','description','entity']
    icd_entity_df.drop(['description'],axis=1,inplace=True)
    print(icd_entity_df.head(5))
    #去重
    icd_entity_df.drop_duplicates(['icd'],inplace=True)

    #再次进行合并
    result2=pd.merge(result1,icd_entity_df,how='left',on=['icd'])
    print(len(result2))
    #将结果写入文件
    result2.to_csv('../data/ICD_desc_entity.csv',index=False)

def merge_ehr():
    df_ehr=pd.read_csv('../data/note_labeled.csv')
    df_ehr.columns=['SUBJECT_ID','HADM_ID','TEXT','LABELS']
    print(df_ehr.head(5))

    #读入train 中对应的entity
    df_train=pd.read_csv('../data/disch_train_split_entity.csv')
    df_train.columns=['SUBJECT_ID','HADM_ID','TEXT','LABELS','entity']
    df_train.drop(['TEXT','LABELS'],axis=1,inplace=True)

    # 读入dev 中对应的entity
    df_dev = pd.read_csv('../data/disch_dev_split_entity.csv')
    df_dev.columns = ['SUBJECT_ID', 'HADM_ID', 'TEXT', 'LABELS', 'entity']
    df_dev.drop(['TEXT', 'LABELS'], axis=1, inplace=True)

    # 读入test 中对应的entity
    df_test = pd.read_csv('../data/disch_test_split_entity.csv')
    df_test.columns = ['SUBJECT_ID', 'HADM_ID', 'TEXT', 'LABELS', 'entity']
    df_test.drop(['TEXT', 'LABELS'], axis=1, inplace=True)

    #将三者进行拼接
    df_entity=pd.concat([df_train,df_dev,df_test])
    df_entity.drop_duplicates(inplace=True)

    #合并到原来的EHR中
    result=pd.merge(df_ehr,df_entity,how='left',on=['SUBJECT_ID','HADM_ID'])

    # 将结果写入文件
    result.to_csv('../data/note_labeled_entity.csv', index=False)

if __name__ == '__main__':
    merge_data()
    #merge_ehr()
