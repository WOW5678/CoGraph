# -*- coding:utf-8 -*-
"""
@Time: 2020/09/24 16:14
@Author: Shanshan Wang
@Version: Python 3.7
@Function:
"""
import numpy as np
import os
import argparse

import prepare_dataset
import prototype_model
import gnn
import utils

import torch
import torch.nn.functional as F
import torch.optim as optim
import random


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 参数定义
PARSER=argparse.ArgumentParser(description='The code for the prototype network')

PARSER.add_argument('-datadir','--datadir',default='/data/wangshanshan-slurm/sigir-2021/data')
# 图卷积相关参数
#PARSER.add_argument('-input_dim','--input_dim',default=4095,type=int)
PARSER.add_argument('-hidden_dim_list','--hidden_dim_list',default=[200,100])
#PARSER.add_argument('-fixed_node_num','--fixed_node_num',default=50)

# few-shot learning阶段
PARSER.add_argument('-max_epochs','--max_epochs',default=100,type=int,help='training epochs')
PARSER.add_argument('-max_episode','--max_episode',default=100,type=int,help='episode number in each epoch')
PARSER.add_argument('-n_way','--n_way',default=6,type=int,help='class number of each episode')
PARSER.add_argument('-n_support','--n_support',default=3,type=int,help='sample number of each class')
PARSER.add_argument('-n_query','--n_query',default=2,type=int,help='sample number of each class to construct query')
PARSER.add_argument('-lr','--lr',default=0.00001,type=float,help='learning rate')

# few-shot testing阶段
PARSER.add_argument('-test_n_way','--test_n_way',default=6,type=int,help='class number of each episode')
PARSER.add_argument('-test_n_support','--test_n_support',default=1,type=int,help='sample number of each class')
PARSER.add_argument('-test_n_query','--test_n_query',default=1,type=int,help='sample number of each class to construct query')
PARSER.add_argument('-test_episode','--test_episode',default=1000,type=int,help='episode number in the test phrase')

args=PARSER.parse_args()
args.device=('cuda:0' if torch.cuda.is_available() else 'cpu')
print('args.device:',args.device)

def train(model,optimizer,train_x,train_y,args):
    '''
    :param model:
    :param optimizer:
    :param train_x:
    :param train_y:
    :param args:
    :return:
    '''
    model.train()

    scheduler=optim.lr_scheduler.StepLR(optimizer,1,gamma=0.5,last_epoch=-1)
    epoch=0
    stop=False

    while epoch <args.max_epochs and not stop:
        running_loss=0.0
        running_acc=0.0
        running_prec=0.0
        running_recall=0.0
        running_f1=0.0

        for episode in range(args.max_episode):
            sample=utils.create_sample(train_x,train_y,args)
            optimizer.zero_grad()

            loss,acc,prec,recall,f1=model.set_forward_loss(sample)
            running_loss+=loss.item()
            running_acc+=acc
            running_prec+=prec
            running_recall+=recall
            running_f1+=f1

            loss.backward()
            optimizer.step()

        epoch_loss=(running_loss/args.max_episode)
        epoch_acc=running_acc/args.max_episode
        epoch_prec=running_prec/args.max_episode
        epoch_recall=running_recall/args.max_episode
        epoch_f1=running_f1/args.max_episode

        print('Epoch:{:d}--Loss:{:.4f}--ACC:{:.4f}--Prec:{:.4f}--Recall:{:.4f}--F1:{:.4f}'.format(epoch+1,epoch_loss,epoch_acc,epoch_prec,epoch_recall,epoch_f1))
        epoch+=1
        scheduler.step()

def test(model,test_x,test_y,args):
    '''
    :param model:
    :param optimizer:
    :param train_x:
    :param train_y:
    :param args:
    :return:
    '''
    model.eval()
    model.encoder.eval()
    # scheduler=optim.lr_scheduler.StepLR(optimizer,1,gamma=0.5,last_epoch=-1)
    epoch=0
    stop=False

    while epoch <args.max_epochs and not stop:
        running_loss=0.0
        running_acc=0.0
        running_prec=0.0
        running_recall=0.0
        running_f1=0.0

        for episode in range(args.test_episode):
            sample=utils.create_test_sample(test_x,test_y,args)
            if sample:
                with torch.no_grad():
                    loss,acc,prec,recall,f1=model.set_forward_loss(sample)
                    running_loss+=loss.item()
                    running_acc+=acc
                    running_prec+=prec
                    running_recall+=recall
                    running_f1+=f1

        epoch_loss=(running_loss/args.test_episode)
        epoch_acc=running_acc/args.test_episode
        epoch_prec=running_prec/args.test_episode
        epoch_recall=running_recall/args.test_episode
        epoch_f1=running_f1/args.test_episode

        print('TEST Epoch:{:d}--Loss:{:.4f}--ACC:{:.4f}--Prec:{:.4f}--Recall:{:.4f}--F1:{:.4f}'.format(epoch+1,epoch_loss,epoch_acc,epoch_prec,epoch_recall,epoch_f1))
        epoch+=1

if __name__ == '__main__':
    seed=1234
    seed_everything(seed)
    # step1:加载训练数据集
    EHR_train, ENTITY_set_train=prepare_dataset.get_entities(os.path.join(args.datadir,'background'))
    EHR_test, ENTITY_set_test = prepare_dataset.get_entities(os.path.join(args.datadir, 'evalution'))
    EHR=EHR_train | EHR_test
    #ICD=ICD_set_train+ICD_set_test
    ENTITY=ENTITY_set_train | ENTITY_set_test
    # 将所有的实体进行ID化
    EHR2ID = {e: (id + 1) for id, e in enumerate(EHR)}
    ENTITY2ID = {entity: (id + 1) + len(EHR) for id, entity in enumerate(ENTITY)}
    ALL2ID = dict(EHR2ID, **ENTITY2ID)
    ALL2ID['PAD'] = 0
    args.entity2id=ALL2ID
    args.input_dim=len(ALL2ID)
    print(len(ALL2ID), 'all2id:', ALL2ID)

    graphs,labels,fixed_node_num=prepare_dataset.create_graph_paths(os.path.join(args.datadir,'background'),ALL2ID)
    graphs_test, labels_test, fixed_node_num = prepare_dataset.create_graph_paths(os.path.join(args.datadir, 'evaluation'),ALL2ID)
    args.fixed_node_num = fixed_node_num

    # step2:创建模型
    encoder=gnn.gcn(args)
    encoder.to(args.device)
    model=prototype_model.ProtoNet(args,encoder)
    model.to(args.device)
    optimizer=optim.Adam(model.parameters(),lr=args.lr)

    # step3:训练模型
    train(model, optimizer, graphs,labels, args)
    # step4:测试模型
    test(model, graphs_test,labels_test, args)




