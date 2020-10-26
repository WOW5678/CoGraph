# -*- coding:utf-8 -*-
"""
@Time: 2020/09/24 19:06
@Author: Shanshan Wang
@Version: Python 3.7
@Function:
"""
import numpy as np
import torch
import random
import scipy.sparse as sp
def create_sample(graphs,labels,args):
    """
    :param n_way:
    :param n_support:
    :param n_query:
    :param datax:
    :param datay:
    :return:
    """

    n_way, n_support, n_query=args.n_way,args.n_support,args.n_query
    sample=[]
    nodes_sample=[]
    # 随机选取n_way个class
    k=random.sample(set(labels),n_way)

    for cls in k:
        # 筛选出该cls对应的样本数据
        datax_cls=[]
        #datax_cls=graphs[labels==cls]
        # 对筛选出的数据进行随机打乱
        #perm=np.random.permutation(datax_cls)
        for i,label in enumerate(labels):
            if label==cls:
                datax_cls.append(graphs[i])

        # # 从数据中随机选择出n_support+n_query个样本
        sample_cls=random.sample(datax_cls,n_support+n_query)

        sample.append(np.stack([s[0] for s in sample_cls],0))
        nodes_sample.append(np.stack([s[1] for s in sample_cls],0))

    sample=np.stack(sample,0)
    nodes_sample=np.stack(nodes_sample,0)
    #print('sample.shape:',sample.shape)
    return ({'graphs':sample,
             'graph_nodes':nodes_sample,
             'n_way':n_way,
             'n_support':n_support,
             'n_query':n_query,
             })








