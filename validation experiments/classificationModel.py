# -*- coding:utf-8 -*-
"""
@Time: 2020/10/26 20:52
@Author: Shanshan Wang
@Version: Python 3.7
@Function:
"""
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self,encoder):
        super(Classifier, self).__init__()
        self.encoder=encoder
        self.fc=nn.Linear(200,1)

    def forward(self,sample):

       target_inds = torch.arange(0, n_way).view(n_way, 1, 1).expand(n_way, n_query, 1).long().to(self.args.device)
       z=self.encoder(sample)
       output=self.fc(z)

       # 计算概率
       log_p_y = F.log_softmax(output, dim=1)  # [n_way,n_query,n_way]

       loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()  # scalar
       _, y_hat = log_p_y.max(2)  # y_hat:[n_way,n_query] ,_:[n_way,n_query]
       acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()  # scalar

       return loss_val, {'loss': loss_val.item(), 'acc': acc_val.item(), 'y_hat': y_hat}