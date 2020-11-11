# -*- coding:utf-8 -*-
"""
@Time: 2020/09/29 19:07
@Author: Shanshan Wang
@Version: Python 3.7
@Function: 图卷积网络
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class gcn(nn.Module):
    def __init__(self,args, bias=True):  # X_size = num features
        super(gcn, self).__init__()

        self.weight = nn.parameter.Parameter(torch.FloatTensor(args.input_dim, args.hidden_dim_list[0]).to(args.device))  # [7767,330]
        var = 2. / (self.weight.size(1) + self.weight.size(0))
        self.weight.data.normal_(0, var)
        self.weight2 = nn.parameter.Parameter(torch.FloatTensor(args.hidden_dim_list[0], args.hidden_dim_list[1]).to(args.device))  # [330,130]
        var2 = 2. / (self.weight2.size(1) + self.weight2.size(0))
        self.weight2.data.normal_(0, var2)
        if bias:
            self.bias = nn.parameter.Parameter(torch.FloatTensor(args.hidden_dim_list[0]).to(args.device))
            self.bias.data.normal_(0, var)
            self.bias2 = nn.parameter.Parameter(torch.FloatTensor(args.hidden_dim_list[1]).to(args.device))
            self.bias2.data.normal_(0, var2)
        else:
            self.register_parameter("bias", None)
        #self.fc1 = nn.Linear(args.hidden_dim_list[1], args.num_classes)  # [130,66]
        entities=torch.tensor([i for i in range(len(args.entity2id))]).unsqueeze(1).to(args.device)
        self.node_features = torch.FloatTensor(len(args.entity2id), len(args.entity2id)).zero_().to(args.device)
        self.node_features = self.node_features.scatter_(dim=1, index=entities, value=1)

        self.args=args
    def forward(self,A_hat,X):  ### 2-layer GCN architecture
        A_hat = torch.tensor(A_hat, requires_grad=False).float().to(self.args.device)
        X=X.view(X.size()[0]*X.size()[1]).long()
        X=self.node_features[X]
        X = torch.matmul(X, self.weight) #[540,200]
        if self.bias is not None:
            X = (X + self.bias)
        X=X.view(A_hat.size()[0],A_hat.size()[1],-1) #[18,30,200]
        X = F.relu(torch.bmm(A_hat, X)) #[18,30,200]
        X=X.view(X.size()[0]*X.size()[1],-1)
        X = torch.mm(X, self.weight2)
        if self.bias2 is not None:
            X = (X + self.bias2)
        X = X.view(A_hat.size()[0], A_hat.size()[1], -1)  # [18,30,200]
        X = F.relu(torch.bmm(A_hat, X))
        #return self.fc1(X)
        return torch.mean(X,1)