# -*- coding:utf-8 -*-
"""
@Time: 2020/09/24 16:06
@Author: Shanshan Wang
@Version: Python 3.7
@Function: 模型部分
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProtoNet(nn.Module):
    def __init__(self,args,encoder):
        super(ProtoNet, self).__init__()
        self.encoder=encoder  # encoder 是图卷积网络 对每个样本进行向量化
        self.args=args

    def set_forward_loss(self,sample):
        '''
        :param sample:
        :return:
        '''
        sample__=torch.Tensor(sample['graphs']).to(self.args.device).long()   #[6,5,2542]
        n_way=sample['n_way']
        n_support=sample['n_support']
        n_query=sample['n_query']

        sample_support=sample__[:,:n_support] #[n_way,n_support, 30,30]
        sample_query=sample__[:,n_support:] #[n_way,n_query,30,30]


        target_inds=torch.arange(0,n_way).view(n_way,1,1).expand(n_way,n_query,1).long().to(self.args.device)


        sample_ = torch.cat([sample_support.contiguous().view(n_way * n_support, *sample_support.size()[2:]),
                       sample_query.contiguous().view(n_way * n_query, *sample_query.size()[2:])],0)  # [n_way*(n_support+n_query),30,30]

        # 对图执行编码
        z = self.encoder.forward(sample_)  # [n_way*(n_support+n_query),64]

        z_dim = z.size(-1)  # 64
        z_proto = z[:n_way * n_support].view(n_way, n_support, z_dim).mean(1)  # [n_way,64]
        z_query = z[n_way * n_query:]  # (n_way*n_query,64)

        # 计算距离
        dists = euclidean_dist(z_query, z_proto)  # [n_way*n_query,n_way]

        # 计算概率
        log_p_y = F.log_softmax(-dists, dim=1).view(n_way, n_query, -1)  # [n_way,n_query,n_way]

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()  # scalar
        _, y_hat = log_p_y.max(2)  # y_hat:[n_way,n_query] ,_:[n_way,n_query]
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()  # scalar

        return loss_val, {'loss': loss_val.item(), 'acc': acc_val.item(), 'y_hat': y_hat}

def euclidean_dist(x,y):
    '''
    :param x: query sample
    :param y: class prototype
    :return:
    '''
    n=x.size(0)
    m=y.size(0)
    d=x.size(1)
    assert d==y.size(1)

    x=x.unsqueeze(1).expand(n,m,d) # x.unsqueeze(1):(n,1,d)
    y=y.unsqueeze(0).expand(n,m,d) # y.unsqueeze(1):(1,m,d)
    return torch.pow(x-y,2).sum(2) # (n,m)

