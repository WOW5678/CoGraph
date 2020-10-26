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

import prototype_model
import cnn_model
import process_raw
import utils

import torch
import torch.nn.functional as F
import torch.optim as optim
import random
from classificationModel import  Classifier

#设置随机种子
seed=2021
#random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# 参数定义
PARSER=argparse.ArgumentParser(description='The code for the prototype network')

PARSER.add_argument('-datadir','--datadir',default='/home/wangshanshan/sigir-2021/data')
# 图卷积相关参数
PARSER.add_argument('-input_dim','--input_dim',default=4095,type=int)
PARSER.add_argument('-hidden_dim_list','--hidden_dim_list',default=[200,100])
#PARSER.add_argument('-fixed_node_num','--fixed_node_num',default=50)

# few-shot learning阶段
PARSER.add_argument('-max_epochs','--max_epochs',default=1000,type=int,help='training epochs')
PARSER.add_argument('-max_episode','--max_episode',default=100,type=int,help='episode number in each epoch')
PARSER.add_argument('-n_way','--n_way',default=6,type=int,help='class number of each episode')
PARSER.add_argument('-n_support','--n_support',default=3,type=int,help='sample number of each class')
PARSER.add_argument('-n_query','--n_query',default=2,type=int,help='sample number of each class to construct query')
PARSER.add_argument('-lr','--lr',default=0.0001,type=float,help='learning rate')

# few-shot testing阶段
PARSER.add_argument('-test_n_way','--test_n_way',default=6,type=int,help='class number of each episode')
PARSER.add_argument('-test_n_support','--test_n_support',default=5,type=int,help='sample number of each class')
PARSER.add_argument('-test_n_query','--test_n_query',default=5,type=int,help='sample number of each class to construct query')
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

        for episode in range(args.max_episode):
            sample=utils.create_sample(train_x,train_y,args)
            optimizer.zero_grad()

            loss,output=model.set_forward_loss(sample)
            running_loss+=output['loss']
            running_acc+=output['acc']

            loss.backward()
            optimizer.step()

        epoch_loss=(running_loss/args.max_episode)
        epoch_acc=running_acc/args.max_episode
        print('Epoch:{:d}--Loss:{:.4f}--ACC:{:.4f}'.format(epoch+1,epoch_loss,epoch_acc))
        epoch+=1
        scheduler.step()


if __name__ == '__main__':
    # step1:加载训练数据集
    # if os.path.exists():
    #     pass
    # else:
    data,labels,word2ix, ix2word=process_raw.get_data(os.path.join("/home/wangshanshan/sigir-2021/data",'background'))

    args.word2ix=word2ix
    args.ix2word=ix2word

    # step2:创建模型
    encoder=cnn_model.EHREncoder(args)
    encoder.to(args.device)
    model=prototype_model.ProtoNet(args,encoder)
    model.to(args.device)
    optimizer=optim.Adam(model.parameters(),lr=args.lr)

    # step3:训练模型
    train(model, optimizer,data,labels, args)
    # step4:测试模型
    #test()




