# -*- coding:utf-8 -*-
'''
Create time: 2020/4/10 13:55
@Author: 大丫头
'''
import csv
import tagme
import requests
import pickle

import warnings
warnings.filterwarnings('ignore')

print(requests.__version__)

requests.adapters.DEFAULT_RETRIES = 2  # 增加重连次数
tagme.GCUBE_TOKEN="9e16d353-0c47-4fef-8ccc-676a8591f478-843339462"

proxies = {
  "https": "127.0.0.1:8001"
}

# 从电子病历中提取实体
def extract_entity(filename,label_desc):
    '''
    :param filename:电子病历文件
    :param label_desc:ICD以及对应的描述（dict）
    :return:
        label_entity.pkl 文件：每个label对应的实体list（dict）
        EHR-label-entity-kg.csv文件：相比filename增加了EHR对应的实体这一列
    '''
    # 结果写入文件
    writer_f=open('data/EHR-label-entity-kg.csv','w',newline='')
    writer=csv.writer(writer_f)
    label_entity_EHR_related={}
    #打开文件
    with open(filename,'r') as f:
        reader=csv.reader(f)
        data=[row for row in reader][1:]
        for row in data:
            # 原数据：SUBJECT_ID,HADM_ID,TEXT,LABELS
            print('row:',row)
            count=0
            # 利用tagme提取其中的实体
            #row='group of metabolic disorders characterized by high blood sugar levels over a prolonged period'
            try:
                tomatoes_mentions = tagme.mentions(row[2])
                mentions=tomatoes_mentions.mentions
                # 将数据写入文件中
                content=[row[0],row[1],row[2],0,row[3]]
                content[3]=';'.join([mention.__str__().strip().split('[')[0][:-1] for mention in mentions])
                if len(content)>0:
                    writer.writerow(content)
                    count+=1
            except:
                pass
            labels=row[3].split(';')
            for label in labels:
                if label in label_desc:
                    desc = label_desc.get(label)
                    try:
                        tomatoes_mentions = tagme.mentions(desc)
                        mentions = tomatoes_mentions.mentions
                        if label not in label_entity_EHR_related:
                            label_entity_EHR_related[label]=[mention.__str__().strip().split('[')[0][:-1] for mention in mentions]
                    except:
                        label_entity_EHR_related[label] = []
                else:
                    label_entity_EHR_related[label] = []
                print(label_entity_EHR_related.get(label))

    writer_f.close()
    # 将label_entity_EHR_related保存下来 以备后面的使用
    with open('data/label_entity.pkl','wb') as f:
        pickle.dump(label_entity_EHR_related,f)


def get_label_desc(ICD_desc_file):
    '''
    :param ICD_desc_file: ICD的描述文件即ICD9_descriptions.txt
    :return:label_desc:每个ICD对应的描述（dict）
    '''
    label_desc={}
    # 打开label描述文件
    with open(ICD_desc_file,'r') as f:
        lines=f.readlines()
        for line in lines:
            print(repr(line))
            label,desc=line[:-1].split('\t')
            if label not in label_desc:
                label_desc[label]=desc
    return label_desc

if __name__ == '__main__':
    # step1: extract entity
    label_desc=get_label_desc('data/ICD9_descriptions.txt')
    extract_entity('data/note_labeled.csv',label_desc)

