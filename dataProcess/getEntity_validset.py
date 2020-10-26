# -*- coding:utf-8 -*-
"""
@Time: 2019/06/25 20:18
@Author: Shanshan Wang
@Version: Python 3.7
@Function: 从电子病历数据集中提取医疗实体
"""
from dataProcess.pyMap import MetaMap
import pandas as pd
from utils import xml_to_soup, extract_results_from_soup
import csv
def ehr_parse(file,output_file):
    #将处理好的实体写入到一个文件中
    wf=open(output_file,'w',newline='')
    writer=csv.writer(wf)


    with open(file,'r') as f:
        reader=csv.reader(f)
        next(reader)
        #记录处理了多少个文件
        i=0
        for row in reader:
            print(row[2])
            result=MM.process_text(row[2],r'metamap14.bat --XMLf')
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
                print('items:',items)

                #df.to_csv(output_file,mode='a',index=False,header=False)
                if len(items)>0:
                    writer.writerow([row[0],row[1],items,row[3]])
            print('第%d个文件处理完毕！'%i)
            i+=1
            #print(df.columns)
            #print(df[ 'SemanticType'])
            #print(df[ 'SemanticType'])

if __name__ == '__main__':

    MM=MetaMap()
    MM.start_skrmedpost_server(r'skrmedpostctl_start.bat')
    ehr_parse('../data/disch_dev_split.csv','../data/entity_dev.csv')


