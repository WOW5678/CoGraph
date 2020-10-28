# -*- coding:utf-8 -*-
"""
@Time: 2020/10/12 14:49
@Author: Shanshan Wang
@Version: Python 3.7
@Function:
"""
import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader

class EHRDataset(Dataset):

    def __init__(self):
        pad_data, labels, word2ix, ix2word = get_data(os.path.join("data", 'background'))
        self.x=pad_data
        self.y=labels
        self.word2ix=word2ix
        self.ix2word=ix2word

    def __getitem__(self, index):
        return self.x[index],self.y[index]

    def __len__(self):
        return len(self.x)



def get_data(data_dir):
    data=[]
    labels=[]
    for root,dirs,files in os.walk(data_dir):
        for file in files:
            file_path=os.path.join(root,file)
            # 打开file_path文件
            #print('file_path:',file_path)
            label_cls = root.split('/')[-1]
            with open(file_path,'r') as f:
                reader=csv.reader(f)
                data_cls = [row[2].split() for row in reader]
                data=data+data_cls
                labels=labels+[label_cls]*len(data_cls)
    print('data:',len(data))
    print('labels:',len(labels))

    # word字典
    words={word for sen in data for word in sen}
    word2ix={w:(ix+1) for ix,w in enumerate(words)}
    word2ix['<PAD>']=0
    ix2word={ix:w  for w,ix in word2ix.items()}
    new_data=[[word2ix[w] for w in s] for s in data]
    # pad 和截断EHR
    pad_data=pad_ehr(new_data,maxlen=None,padding='pre',truncating='pre',value=word2ix['<PAD>'])

    return pad_data,labels,word2ix,ix2word



def pad_ehr(sequences,maxlen=None,dtype='int32',padding='pre',truncating='pre',value=0.):
    '''
    :param data:
    :param maxlen:
    :param padding:
    :param truncating:
    :param value:
    :return:
    '''
    if not hasattr(sequences,'__len__'):
        raise ValueError('sequences must be iterable.')
    lengths=[]
    for seq in sequences:
        if not hasattr(seq,"__len__"):
            raise ValueError('sequence must be a list of iterable. Found non-iterable:'+str(seq))
        lengths.append(len(seq))

    num_samples=len(sequences)
    if maxlen is None:
        maxlen=np.max(lengths)
    sample_shape=tuple()
    for s in sequences:
        if len(s)>0:
            sample_shape=np.asarray(s).shape[1:]
            break
    x=(np.ones((num_samples,maxlen)+sample_shape)*value).astype(dtype)
    for idx,s in enumerate(sequences):
        if not len(s):
            continue
        if truncating=='pre':
            trunc=s[-maxlen:]
        elif truncating=='post':
            trunc=s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understand.'%truncating)

        # check the trunc has expected shape
        trunc=np.asarray(trunc,dtype=dtype)
        if trunc.shape[1:]!=sample_shape:
            raise ValueError('shape of sample %s of sequences at position %s is different from the expected shape %s'%(trunc.shape[1:],idx,sample_shape))
        if padding=='post':
            x[idx,:len(trunc)]=trunc
        elif padding=='pre':
            x[idx,-len(trunc):]=trunc
        else:
            raise ValueError('Padding type %s not understand.'%padding)
    return x
