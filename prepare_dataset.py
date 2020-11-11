# -*- coding:utf-8 -*-
"""
@Time: 2020/09/22 9:29
@Author: Shanshan Wang
@Version: Python 3.7
@Function: 将数据集规整成few-shot learnaing的格式
"""
import csv
import pickle
import os
import numpy  as np
import scipy as sp
import random

def split_class(filename,sample_min_number):
    '''
    当ICD对应的样本个数>=sample_min_number时，则将ICD加入tran_class
    :param filename: EHR文件
    :param sample_min_number: 每个ICD对应的样本最小个数
    :return:
        train_class_number:作为训练集的class, dict
        test_class_number:作为测试集的class，dict
    '''

    # 打开数据集
    #SUBJECT_ID, HADM_ID, TEXT, Entities,LABELS
    label_number=dict()
    with open(filename,'r') as f:
        reader=csv.reader(f)
        data=[row[4] for row in reader][1:]
        for row in data:
            labels=row.split(';')
            for label in labels:
                if label not in label_number:
                    label_number[label]=1
                else:
                    label_number[label]+=1
    # 对label_number按照频率进行排序
    # sorted(label_number,key=lambda x:x[1],reverse=True)
    train_class_number,test_class_number={},{}
    # 选择频率超过sample_min_number的class为train class，剩余的为test class
    print(label_number)
    for label,number in label_number.items():
        if number>=sample_min_number:
            train_class_number[label]=number
        else:
            test_class_number[label]=number

    # 统计下train class与test class的个数
    print('train class number:',len(train_class_number))
    print('test class number:',len(test_class_number))
    return train_class_number,test_class_number

def regular_files(filename,train_class_number,test_class_number):
    '''
    将每个class对应的样本的样本写入到对应的文件中

    :param label_entity_pickle_file:保存着每个ICD对应的实体（dict）
    :param filename:EHR文件（含EHR的实体）
    :param train_class_number:作为训练集的class(dict,value为样本个数)
    :param test_class_number:作为测试集的class(dict,value为样本个数)
    :return:
    '''
    #加载处理好的label实体文件
    # with open(label_entity_pickle_file,'rb') as f:
    #     label_entity=pickle.load(f)
    remove_last_dirs()
    with open(filename,'r') as f:
        reader=csv.reader(f)
        data=[row for row in reader][1:]
        for row in data:
            labels=row[4].split(';')
            for label in labels:
                if label in train_class_number:
                    if not os.path.exists('data/background/%s'%(label)):
                        os.makedirs('data/background/%s'%(label))
                    # 打开这个新创建的csv文件 并将改行数据写入进去
                    with open('data/background/%s/samples_%s.csv'%(label,label),'a+',newline='') as f_w:
                        writer = csv.writer(f_w)
                        writer.writerow([row[0],row[1],row[2],row[3],label])

                elif label in test_class_number:
                    if not os.path.exists('data/evaluation/%s' % (label)):
                        os.makedirs('data/evaluation/%s' % (label))
                    # 打开这个新创建的csv文件 并将改行数据写入进去
                    with open('data/evaluation/%s/samples_%s.csv' % (label, label), 'a+',newline='') as f_w:
                        writer = csv.writer(f_w)
                        writer.writerow([row[0], row[1], row[2], row[3], label])
                else:
                    print('There are some errors.')

def remove_last_dirs():
    if os.path.exists('data/background'):
        os.rmdir('data/background')

def get_entities(data_dir):
    # step1:先要统计出所有的实体个数，实体类型包括EHR,entities, ICD
    EHR_set, ICD_set, ENTITY_set = set(), set(), set()

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            # 打开file_path文件
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                data = [row for row in reader]
                for row in data:
                    # EHR_set.add('E_'+row[1])
                    # ICD_set.add('I_'+row[4])
                    EHR_set.add(row[1])
                    ICD_set.add(row[4])
                    for item in row[3].split(';'):
                        ENTITY_set.add(item)
            print('root:', root)

    print(len(EHR_set), EHR_set)
    print(len(ICD_set), ICD_set)
    print(len(ENTITY_set), ENTITY_set)
    return EHR_set,ENTITY_set

def create_graph_paths(data_dir,ALL2ID):
    '''
    将每个样本转换为graph
    :param data_dir:样本所在的文件夹
    :param label_entity_pickle_file: ICD以及对应的实体（dict）
    :return:
    '''
    fixed_node_num=300
    # step3: 建立graph
    graphs=[]
    nodes=[]
    labels=[]
    for root,dirs,files in os.walk(data_dir):
        for file in files:
            file_path=os.path.join(root,file)
            label = root.split('/')[-1]

            # 打开file_path文件
            with open(file_path,'r') as f:
                reader=csv.reader(f)
                data = [row for row in reader]
                for row in data:
                    EHR_entities=row[3].split(';')
                    # 每行数据为一个样本 都要建立一个小图
                    row_graph=connect_nodes(row[1],EHR_entities)
                    # 将row_graph进行ICD化
                    row_graph_id={}
                    for key,value in row_graph.items():
                        if key in ALL2ID:
                            key_id=ALL2ID.get(key)
                            row_graph_id[key_id]=[ALL2ID.get(item) for item in value]
                        else:
                            print('There are some errors.')
                    print('row_graph_len:',len(row_graph_id))

                    # if len(row_graph_id)>fixed_node_num:
                    #     fixed_node_num=len(row_graph_id)

                    #每一行都会组建一个图 但由于数据的限定 会随机从中选择固定的node个数
                    row_adj, row_nodes=pad_graph(row_graph_id,fixed_node_num)
                    graphs.append(row_adj)
                    nodes.append(row_nodes)
                    labels.append(label)
    print('graph_number:',len(graphs))
    print('label_number:',len(labels))
    graph_samples=list(zip(graphs,nodes))

    return graph_samples,labels,fixed_node_num



def connect_nodes(EHR,EHR_E):
    '''
    :param EHR: EHR 节点
    :param EHR_E: EHR中包含的实体节点
    :param ICD_E: ICD描述中包含的实体节点
    :param ICD: ICD 节点
    :return: dict
    '''
    # step1: 在EHR_E与ICD_E之间建立全连接
    #ALL_E=set(EHR_E) | set(ICD_E) #集合求并集
    ALL_E=set(EHR_E)
    graph_dict={}
    for e in ALL_E:
        if e not in graph_dict:
            graph_dict[e]=list(ALL_E - set(e)) # 集合求差集

    # step2: 增加EHR 与 ICD与实体间的连接
    graph_dict[EHR]=EHR_E
    #graph_dict[ICD]=ICD_E

    # step3: 增加EHR ICD中每个实体与EHR ICD的联系
    for e in  set(EHR_E):
        if e in graph_dict:
            graph_dict[e].append(EHR)
        else:
            print('Some errors.')
    # for e in set(ICD_E):
    #     if e in graph_dict:
    #         graph_dict[e].append(ICD)
    #     else:
    #         print('Some errors.')
    return graph_dict

def pad_graph(graph,fixed_node_num):
    nodes=list(graph.keys())
    if len(nodes)<fixed_node_num:
        nodes=nodes+[0]*(fixed_node_num-len(nodes))
    else:
        nodes=random.sample(nodes,fixed_node_num)
    # 给每个nodes 分配一个子图id 因为要创建邻接矩阵
    subnode2id = {n: id for id, n in enumerate(nodes)}
    # crete graphs
    adjList=[]
    adjMatrix=np.zeros((fixed_node_num,fixed_node_num))
    for node,neighbor in graph.items():
        for neigh in neighbor:
            if neigh in subnode2id and node in subnode2id:
                adjList.append([subnode2id.get(node),subnode2id.get(neigh)])
                adjMatrix[subnode2id.get(node)][subnode2id.get(neigh)]=1

    #adjList=np.array(adjList)
    #print(adjList.shape)
    #sparse_adj=sp.coo_matrix((np.ones(len(adjList)),(adjList[:,0],adjList[:,1])),shape=(fixed_node_num,fixed_node_num),dtype=np.float32)
    return adjMatrix,nodes

if __name__ == '__main__':
    train_class_number,test_class_number=split_class('data/EHR-label-entity-kg.csv',sample_min_number=5)
    regular_files('data/EHR-label-entity-kg.csv',train_class_number,test_class_number)
    #create_graph_paths('data\\background')



