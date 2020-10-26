# -*- coding:utf-8 -*-
"""
@Time: 2019/06/26 15:57
@Author: Shanshan Wang
@Version: Python 3.7
@Function: 针对每个ICD代码，先找见它对应的描述信息，然后根据描述信息找到最相似的wikipedia页面，之后使用metamap工具提取wikipedia页面中的
医疗实体
"""
from dataProcess.pyMap import MetaMap
import pandas as pd
from utils import xml_to_soup, extract_results_from_soup
import csv
import wikipedia

def get_wikipedia_content(filename):
    wf=open('../data/ICD_descript_entity.csv','w',newline='')
    writer=csv.writer(wf)

    wfailf=open('../data/failICDs.csv','w',newline='')
    failWriter=csv.writer(wfailf)

    #获取使用的数据集集中所有的ICD code
    codes=[]
    with open(filename,'r') as f:
        reader=csv.reader(f)
        #next(reader)
        for row in reader:
            codes.extend(row[3].split(';'))
    codes=set(codes)
    print('len codes:',len(codes))
    #获取每个ICD的描述信息，字典形式，key为ICD，value为对应的描述
    icddict={}
    with open('../data/ICD9_descriptions.txt','r') as f:
        for line in f.readlines():
            #print(repr(line))
            lineList=line[:-1].split('\t')
            if lineList[0] in codes and lineList[0] not in icddict:
                icddict[lineList[0]]=lineList[1]
    print(icddict)
    #将字典中的每个描述 作为weikipedia页面的查询
    failCount=0
    allCount=0
    for key,value in icddict.items():
        #try:
        #content=wikipedia.summary('diabetes')
        print('key:',key)
        #print('content:',content)
        #使用metamap工具提取content中出现的实体
        content='Injury due to legal intervention by other specified means'
        items=ehr_parse(content)
        if len(items)>0:
            print(items)
            #将ICD code,描述，items写入到文件中
            writer.writerow([key,value,items])
        #except:
        print('key:%s fails. its content:%s'%(key,value))
        failCount+=1
        print('第 %d failed.'%failCount)
        #将失败的icd写入到文件中
        failWriter.writerow([key,value])

        allCount+=1
        print('进行到 %d:'%allCount)
    wf.close()
    print('allCount:%d,failCount:%d'%(allCount,failCount))

def ehr_parse(content):
        result=MM.process_text(content,r'E:/software/public_mm_win32_main_2014/public_mm/bin//metamap14.bat --XMLf')
        soup = xml_to_soup(result)
        out, extra_metadata = extract_results_from_soup(soup)
        df = pd.DataFrame(out)
        #print(df)
        if df.empty==False:
            #仅保留那些特定的字段 标准表达和语义类别以及否定标识
            df.drop(columns=['FullLexicalUnit', 'Word', 'LexicalCategory', 'CUI'],inplace=True)
            #仅保留有特定实体的行
            #疾病或综合征，病理功能，诊断过程，检验过程，药理物质，肿瘤过程,治疗或预防措施，症状,诊断过程,解剖异常
            df = df[df['SemanticType'].isin(['dsyn', 'patf', 'diap','lapr','phsu','neop','topp','sosy','diap','anab'])]
            # 判断该实体对应的否定是否为true 若是否定的，则也不保留该实体
            df = df[df['Negated'].isin(['0'])]
            df = df.drop_duplicates(['SemanticType', 'PreferredTerm', 'Negated'])
            print('df:',df)
            #将满足条件的行以及需要的字段写入到文件中
            items=''
            for row_ in df.itertuples():
                # print(row)
                # print(row[2])
                items = items + row_[2] + '\t'
            #print('items:',items)
            return items
        return

if __name__ == '__main__':

    MM=MetaMap()
    MM.start_skrmedpost_server(r'E:/software/public_mm_win32_main_2014/public_mm/bin/skrmedpostctl_start.bat')
    get_wikipedia_content('../data/note_labeled.csv')

