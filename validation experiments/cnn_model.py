# -*- coding:utf-8 -*-
"""
@Time: 2020/10/07 15:55
@Author: Shanshan Wang
@Version: Python 3.7
@Function:  用CNN对EHR本文进行编码
"""
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


class EHREncoder(nn.Module):
    def __init__(self,args):
        super(EHREncoder, self).__init__()

        self.args = args
        chanel_num = 1
        #filter_num = self.args.num_filter_maps
        filter_num=24
        dropout_rate=0.3
        cnn_embedding_size=100
        filter_sizes = [3, 4, 5]

        self.embedding = nn.Embedding(len(args.word2ix),cnn_embedding_size)
        self.conv1 = nn.ModuleList(
            [nn.Conv2d(chanel_num, filter_num, (filter_size, cnn_embedding_size)) for filter_size in
             filter_sizes])
        self.dropout1 = nn.Dropout(dropout_rate)

    def forward(self,x):
        # 将每个电子病历转化成Tensor对象
        x = self.embedding(x)
        # print('x:',x.shape)
        x = x.unsqueeze(1)  # [batch_size,1,200,emb_size]
        # print('x_unsequeeze:',x.shape)
        # print(F.relu(self.convs[0](x)).shape) # 每个不同的filter卷积之后为：[batch_size,32,198,1],[batch_size,32,197,1],[batch_size,32,196,1]
        # print('x:',x.shape) #[49, 1, 100]
        # 多通道的图卷积操作（单层卷积）
        x = [F.relu(conv(x), inplace=True) for conv in self.conv1]

        x_ = [item.squeeze(3) for item in x]
        x_ = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x_]
        x_ = torch.cat(x_, 1)
        x_ = self.dropout1(x_)
        return x_



