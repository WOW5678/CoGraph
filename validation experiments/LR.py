# -*- coding:utf-8 -*-
"""
@Time: 2019/07/06 14:48
@Author: Shanshan Wang
@Version: Python 3.7
@Function:
"""

import os
import pickle
import pandas  as pd
import numpy as np
import csv
from string import punctuation
from nltk.corpus import stopwords
from collections import Counter

#import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from full_eval import full_evaluate
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset
from my_pytorchtools import EarlyStopping
import full_eval

import warnings
import numpy as np
from sklearn import linear_model, datasets
warnings.filterwarnings("ignore")
def processEHR(str):
    #对字符串进行预处理操作
    str=str.lower() #全部转化为小写
    str=' '.join([word for word in str.split() if word not in stopwords.words('english')]) #删除停用词
    str=' '.join([word for word in str.split() if word not in  punctuation]) #删除标点符号
    str=' '.join([word for word in str.split() if word.isalpha()]) #移除数字
    str=' '.join([word for word in str.split() if len(word)>=2])
    return str

def analysisLengthDist(patients):
    lengths=[len(x) for x in patients]
    pd.Series(lengths).hist()
    plt.show()
    print('describe information:',pd.Series(lengths).describe())

#根据指定长度对ehrs进行padding或者截取操作
def paddingEHRs(patientEHRIDs,seq_length):
    seqs=np.zeros((len(patientEHRIDs),seq_length),dtype=int)
    for i, ehr in enumerate(patientEHRIDs):
        ehr_len=len(ehr)
        if ehr_len<=seq_length: #需要padding
            zeroes=list(np.zeros(seq_length-ehr_len))
            new=ehr+zeroes
        elif ehr_len>seq_length:
            new=ehr[:seq_length]
        seqs[i,:]=np.array(new)
    return seqs

def parseICDs(datafolder):
    icdsL=[]
    with open(os.path.join(datafolder, 'filter_top_%d_sample50000.csv'%labelNum), 'r') as f:
        reader = csv.reader(f)
        data = [row for row in reader]
        for row in data:
            if row[3] not in icdsL:
                icdsL.extend(row[3].split(';'))
    icdsL=list(set(icdsL))
    return icdsL

def parseEHRs(datafolder,icd2id):
    patientDescribs = []
    patientLabels=[]
    patientWords = []  # 非结构化数据中所有的单词
    with open(os.path.join(datafolder, 'filter_top_%d_sample50000.csv'%labelNum)) as f:
        reader = csv.reader(f)
        data = [row for row in reader]
        for row in data:
            labelList = row[3].split(';')
            str = processEHR(row[2])
            patientDescribs.append(str)
            patientWords.extend(str.split())
            # print('labelList:',labelList)
            # 即过滤了ICD同时将ICD转换为id
            patientLabels.append([icd2id.get(item) for item in labelList if item in icd2id])
    return patientDescribs,patientLabels,patientWords


def label_one_hot(patientLabels,icd2id):
    # 将labels从ID转换为multi-hot编码
    labels = []
    for row in patientLabels:
        temp = np.zeros(len(icd2id))
        temp[row] = 1
        labels.append(temp)
    return labels

class Logistic(nn.Module):
    def __init__(self,vocab_size,emb_size,hidden_size,output_size):
        super(Logistic, self).__init__()
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.embedding = nn.Embedding(embedding_dim=emb_size, num_embeddings=vocab_size+2)
        self.linear1=nn.Linear(emb_size*2000,hidden_size)
        self.linear2=nn.Linear(hidden_size,output_size)
        #self.loss=nn.BCEWithLogitsLoss()
        self.optimizer=torch.optim.Adam(self.parameters(),lr=1e-5)

    def forward(self,x,y_target):
        input=self.embedding(x)
        #对x 进行变形处理
        input=torch.Tensor.view(input,(input.shape[0],input.shape[1]*input.shape[2]))

        input=self.linear1(input)
        output=F.sigmoid(self.linear2(input))
        #print('output.shape:',output.shape)
        #print(output.shape, y_target.shape)
        weights=self.getWeights(y_target)
        loss = F.binary_cross_entropy(output, y_target,weights)

        #print('output:',output)
        preds = (output > 0.5)
        preds=preds.detach().cpu().numpy()
        preds=[np.nonzero(row)[0] for row in preds]
        #print('preds:',preds)
        #output=F.softmax(output)
        #preds=torch.argsort(output,descending=True)[:,:k]

        jaccard,micro_p, macro_p, micro_r, macro_r, micro_f1, macro_f1, micro_auc_roc, macro_auc_roc = full_evaluate(
            preds,y_target.detach().cpu().numpy(),self.output_size)

        return loss,jaccard, micro_p, macro_p, micro_r, macro_r, micro_f1, macro_f1, micro_auc_roc, macro_auc_roc

    def getWeights(self,y):
        #给每个batch的y_target 生成针对这个batch 中每个label的权重
        y=y.cpu().numpy()
        num_1=np.sum(y,axis=1).astype(np.int)
        weights=y.copy()
        #print('num_1 before:',num_1)
        for i in range(len(num_1)):
            index=np.random.choice(y.shape[1],num_1[i]*200)
            #print('index:',index)
            weights[i,index]=1
        # index = np.argwhere(y > 0)
        # weights[index[:, 0], index[:, 1]] = 3
        return torch.Tensor(weights).float().to(device)


def run_training_epoch(train_loader,model,device):
    '''
    :param total_train_batches:
    :return:
    '''
    total_c_loss = 0.0
    total_jaccard=0.0
    total_micro_p = 0.0
    total_macro_p = 0.0
    total_micro_r = 0.0
    total_macro_r = 0.0
    total_micro_f1 = 0.0
    total_macro_f1 = 0.0
    total_micro_auc_roc = 0.0
    total_macro_auc_roc = 0.0


    # icdGraphs=self.data.get_icd_graphs()
    # print('icdGraphs:',icdGraphs)
    model.train()
    for i_batch, sample_batched in enumerate(train_loader):
        x_target, y_target = sample_batched
        x_target = torch.Tensor(x_target).long()
        #y_target = torch.Tensor(y_target).long()

        #print(x_target.shape,y_target.shape) #[32, 200],[32, 8021]
        # if self.isCudaAvailable:
        c_loss, jaccard,micro_p, macro_p, micro_r, macro_r, micro_f1, macro_f1, micro_auc_roc, macro_auc_roc= model(x_target.to(device), y_target.to(device))

        # optimize process
        model.optimizer.zero_grad()
        c_loss.backward()
        model.optimizer.step()

        print(
            "tr_loss:%.4f,jaccard:%.4f,micro_p:%.4f, macro_p:%.4f,micro_r:%.4f, macro_r:%.4f,micro_f1:%.4f, macro_f1:%.4f, micro_auc_roc:%.4f, macro_auc_roc:%.4f," \
             % (c_loss.item(),jaccard,
                                                                                     micro_p,
                                                                                     macro_p,
                                                                                     micro_r,
                                                                                     macro_r,
                                                                                     micro_f1,
                                                                                     macro_f1,
                                                                                     micro_auc_roc,
                                                                                     macro_auc_roc,
                                                                                    ))

        total_c_loss += c_loss.item()
        total_jaccard+=jaccard
        total_micro_p += micro_p
        total_macro_p += macro_p
        total_micro_r += micro_r
        total_macro_r += macro_r

        total_micro_f1 += micro_f1
        total_macro_f1 += macro_f1
        total_micro_auc_roc += micro_auc_roc
        total_macro_auc_roc += macro_auc_roc


    total_c_loss = total_c_loss / len(train_loader)
    total_jaccard=total_jaccard/len(train_loader)
    total_micro_p = total_micro_p / len(train_loader)
    total_macro_p = total_macro_p / len(train_loader)

    total_micro_r = total_micro_r / len(train_loader)
    total_macro_r = total_macro_r / len(train_loader)

    total_micro_f1 = total_micro_f1 / len(train_loader)
    total_macro_f1 = total_macro_f1 / len(train_loader)

    total_micro_auc_roc = total_micro_auc_roc / len(train_loader)
    total_macro_auc_roc = total_macro_auc_roc / len(train_loader)


    # self.scheduler.step(total_c_loss)
    return total_c_loss,total_jaccard, total_micro_p, total_macro_p, total_micro_r, total_macro_r, total_micro_f1, total_macro_f1, total_micro_auc_roc, total_macro_auc_roc

def run_val_epoch(val_loader,model,device):
    '''

    :param total_val_batches:
    :return:
    '''
    total_c_loss = 0.0
    total_jaccard=0.0
    total_micro_p = 0.0
    total_macro_p = 0.0
    total_micro_r = 0.0
    total_macro_r = 0.0
    total_micro_f1 = 0.0
    total_macro_f1 = 0.0
    total_micro_auc_roc = 0.0
    total_macro_auc_roc = 0.0


    model.eval()
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(val_loader):
            x_target, y_target = sample_batched
            x_target = torch.Tensor(x_target).long()
            #y_target = torch.Tensor(y_target).long()

            # if self.isCudaAvailable:
            c_loss,jaccard, micro_p, macro_p, micro_r, macro_r, micro_f1, macro_f1, micro_auc_roc, macro_auc_roc = model(
                 x_target.to(device), y_target.to(device))
            # else:
            #     c_loss,micro_f1, macro_f1, micro_auc_pr, macro_auc_pr, micro_auc_roc, macro_auc_roc, precision_8, precision_40, recall_8, recall_40 = self.matchNN(self.icds, x_target, y_target)

            # iter_out = "val_loss: {}, val_accuracy: {}".format(c_loss.data[0], acc.data[0])
            print(
                "val_loss:%.4f,val_jaccard:%.4f,micro_p:%.4f, macro_p:%.4f,micro_r:%.4f, macro_r:%.4f,micro_f1:%.4f, macro_f1:%.4f, micro_auc_roc:%.4f, macro_auc_roc:%.4f," \
                 % (c_loss.item(),jaccard,
                                                                                         micro_p,
                                                                                         macro_p,
                                                                                         micro_r,
                                                                                         macro_r,
                                                                                         micro_f1,
                                                                                         macro_f1,
                                                                                         micro_auc_roc,
                                                                                         macro_auc_roc))

            total_c_loss += c_loss.item()
            total_jaccard+=jaccard
            total_micro_p += micro_p
            total_macro_p += macro_p
            total_micro_r += micro_r
            total_macro_r += macro_r
            total_micro_f1 += micro_f1
            total_macro_f1 += macro_f1

            total_micro_auc_roc += micro_auc_roc
            total_macro_auc_roc += macro_auc_roc


        total_c_loss = total_c_loss / len(val_loader)
        total_jaccard=total_jaccard/len(val_loader)
        total_micro_p = total_micro_p / len(val_loader)
        total_macro_p = total_macro_p / len(val_loader)

        total_micro_r = total_micro_r / len(val_loader)
        total_macro_r = total_macro_r / len(val_loader)

        total_micro_f1 = total_micro_f1 / len(val_loader)
        total_macro_f1 = total_macro_f1 / len(val_loader)

        total_micro_auc_roc = total_micro_auc_roc / len(val_loader)
        total_macro_auc_roc = total_macro_auc_roc / len(val_loader)


        # self.scheduler.step(total_c_loss)
        return total_c_loss, total_jaccard,total_micro_p, total_macro_p, total_micro_r, total_macro_r, total_micro_f1, total_macro_f1, total_micro_auc_roc, total_macro_auc_roc


def run_test_epoch(test_loader,model,device):
    '''
    :param total_val_batches:
    :return:
    '''
    total_c_loss = 0.0
    total_jaccard=0.0
    total_micro_p = 0.0
    total_macro_p = 0.0
    total_micro_r = 0.0
    total_macro_r = 0.0
    total_micro_f1 = 0.0
    total_macro_f1 = 0.0
    total_micro_auc_roc = 0.0
    total_macro_auc_roc = 0.0


    model.eval()

    with torch.no_grad():
        for i_batch, sample_batched in enumerate(test_loader):
            x_target, y_target = sample_batched
            x_target = torch.Tensor(x_target).long()
            #y_target = torch.Tensor(y_target).long()
            # if self.isCudaAvailable:
            c_loss, jaccard,micro_p, macro_p, micro_r, macro_r, micro_f1, macro_f1, micro_auc_roc, macro_auc_roc = model(
                 x_target.to(device), y_target.to(device))

            print(
                "test_loss:%.4f,jaccad:%.4f,micro_p:%.4f, macro_p:%.4f,micro_r:%.4f, macro_r:%.4f,micro_f1:%.4f, macro_f1:%.4f, micro_auc_roc:%.4f, macro_auc_roc:%.4f," \
                 % (c_loss.item(),jaccard,
                                                                                         micro_p,
                                                                                         macro_p,
                                                                                         micro_r,
                                                                                         macro_r,
                                                                                         micro_f1,
                                                                                         macro_f1,
                                                                                         micro_auc_roc,
                                                                                         macro_auc_roc
                                                                                         ))

            total_c_loss += c_loss.item()
            total_jaccard+=jaccard
            total_micro_p += micro_p
            total_macro_p += macro_p

            total_micro_r += micro_r
            total_macro_r += macro_r

            total_micro_f1 += micro_f1
            total_macro_f1 += macro_f1

            total_micro_auc_roc += micro_auc_roc
            total_macro_auc_roc += macro_auc_roc


        total_c_loss = total_c_loss / len(test_loader)
        total_jaccard=total_jaccard / len(test_loader)

        total_micro_p = total_micro_p / len(test_loader)
        total_macro_p = total_macro_p / len(test_loader)

        total_micro_r = total_micro_r / len(test_loader)
        total_macro_r = total_macro_r / len(test_loader)

        total_micro_f1 = total_micro_f1 / len(test_loader)
        total_macro_f1 = total_macro_f1 / len(test_loader)

        total_micro_auc_roc = total_micro_auc_roc / len(test_loader)
        total_macro_auc_roc = total_macro_auc_roc / len(test_loader)


        # self.scheduler.step(total_c_loss)
        return total_c_loss,total_jaccard, total_micro_p, total_macro_p, total_micro_r, total_macro_r, total_micro_f1, total_macro_f1, total_micro_auc_roc, total_macro_auc_roc

def dataProcess():
    print('Processing data .....')

    icdsL=parseICDs(datafolder)
    print(len(icdsL))
    icd2id = {icd: id for id, icd in enumerate(icdsL)}
    print('icd2id:',len(icd2id))

    patientDescribs,  patientLabels, ehrWords=parseEHRs(datafolder,icd2id)
    patientLabels=label_one_hot(patientLabels,icd2id)

    # 处理非结构化数据
    # 对非结构化的EHR和ICD描述进行预处理
    count_ehrWords = Counter(ehrWords)
    sorted_ehrWords = count_ehrWords.most_common(50000)

    # 将单词映射到字典中
    ehrVocab2id = {w: i + 1 for i, (w, c) in enumerate(sorted_ehrWords)}  # 索引从1开始 因为索引为0的位置为padding字符
    ehrVocab2id['UNK']=len(ehrVocab2id)+1
    #id2ehrVocab = {id: w for w, id in ehrVocab2id_50.items()}
    # 将句子ID化
    patientEHRIDs = []
    for str in patientDescribs:
        r = [ehrVocab2id.get(w,ehrVocab2id.get('UNK')) for w in str.split()]
        patientEHRIDs.append(r)
    # 分析句子的长度 方便选择合适的padding 长度
    #analysisLengthDist(patientEHRIDs)
    # 通过对EHR长度分布的分析，我们将padding长度设置为200，超过200的字符会被截断，不足的会被填充
    # padding EHR
    patientEHRIDs = paddingEHRs(patientEHRIDs,seq_length=2000)
    print('len(patientEHRIDs):{},len(patientLabels):{}'.format(len(patientEHRIDs),len(patientLabels)))

    #将数据进行保存
    with open('../data/patients_labels_%d.pkl'%labelNum,'wb') as f:
        pickle.dump([patientEHRIDs,patientLabels],f)
    with open('../data/ehrVocab2id_%d'%labelNum,'wb') as f:
        pickle.dump(ehrVocab2id,f)

def test_model():
    model=torch.load('model_%d.pkl'%labelNum)
    model.to(device)
    total_test_c_loss, test_jaccard,test_micro_p, test_macro_p, test_micro_r, test_macro_r, test_micro_f1, test_macro_f1, test_micro_auc_roc, test_macro_auc_roc= run_val_epoch(
        test_loader, model, device)
    print('test_loss:{}'.format(total_test_c_loss))
    print('test_jaccard:',test_jaccard)
    print('test_micro_p:', test_micro_p)
    print('test_macro_p:', test_macro_p)
    print('test_micro_r:', test_micro_r)
    print('test_macro_r:', test_macro_r)

    print('test_micro_f1:', test_micro_f1)
    print('test_macro_f1:', test_macro_f1)
    print('test_micro_auc_roc:', test_micro_auc_roc)
    print('test_macro_auc_roc:', test_macro_auc_roc)


if __name__ == '__main__':
    datafolder = '../data'
    print('=================================')
    labelNum=50
    #dataProcess()

   
    #将数据进行保存
    with open('../data/patients_labels_%d.pkl'%labelNum, 'rb') as f:
        patientEHRIDs, patientLabels=pickle.load(f)
    with open('../data/ehrVocab2id_%d'%labelNum, 'rb') as f:
        ehrVocab2id=pickle.load(f)

    #随机分割训练集 测试集 验证集
    train_patients, val_patients, train_labels, val_labels = train_test_split(patientEHRIDs,patientLabels,test_size=(1.0 / 3),random_state=2019)

    val_patients,test_patients, val_labels, test_labels = train_test_split(val_patients,val_labels,test_size=(1.0 / 2),random_state=2019)
    # print(np.array(train_patients).shape,np.array(train_labels).shape)
    # print(np.array(val_patients).shape,np.array(val_labels).shape)
    # print(np.array(test_patients).shape,np.array(test_labels).shape)
    batch_size =32
    total_epochs=100
    device = ("cuda:5" if torch.cuda.is_available() else "cpu")
    best_val_micro_f1=0.0
    # print(len(train_patients[0]))  #
    # print(len(train_labels[0]),len(train_labels[1]))

    # 创建tensor datasets
    train_data = TensorDataset(torch.Tensor(train_patients).float(), torch.Tensor(train_labels).float())
    valid_data = TensorDataset(torch.Tensor(val_patients).float(), torch.Tensor(val_labels).float())
    test_data = TensorDataset(torch.Tensor(test_patients).float(), torch.Tensor(test_labels).float())

    # dataloaders
    # make sure to shuffle your data
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size, drop_last=True)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=True)
    train_batches = len(train_loader)
    val_batches = len(valid_loader)
    test_batches = len(test_loader)

    # 释放内存
    del train_data,valid_data, test_data

    #创建模型 并训练  测试模型
    model=Logistic(vocab_size=len(ehrVocab2id),emb_size=100,hidden_size=128,output_size=labelNum)
    model.to(device)

    # 初始化earlystopping 对象
    early_stopping = EarlyStopping(patience=10, verbose=True)

    # 训练模型
    #with tqdm.tqdm(total=total_train_batches) as pbar_e:
    #针对每个训练epoch
    for e in range(total_epochs):
        total_c_loss,jaccard, micro_p, macro_p, micro_r, macro_r, micro_f1, macro_f1, micro_auc_roc, macro_auc_roc = run_training_epoch(train_loader,model,device)
        print('Epoch:{}:train_loss:{}'.format(e, total_c_loss))
        print('jaccard:', jaccard)
        print('micro_p:', micro_p)
        print('macro_p:', macro_p)
        print('micro_r:', micro_r)
        print('macro_r:', macro_r)
        print('micro_f1:', micro_f1)
        print('macro_f1:', macro_f1)
        print('micro_auc_roc:', micro_auc_roc)
        print('macro_auc_roc:', macro_auc_roc)


        total_val_c_loss, val_jaccard,val_micro_p, val_macro_p, val_micro_r, val_macro_r, val_micro_f1, val_macro_f1, val_micro_auc_roc, val_macro_auc_roc = run_val_epoch(
            valid_loader,model,device)
        print('Epoch:{}: val_loss:{}'.format(e, total_val_c_loss))
        print('val_jaccard:',val_jaccard)
        print('val_micro_p:', val_micro_p)
        print('val_macro_p:', val_macro_p)
        print('val_micro_r:', val_micro_r)
        print('val_macro_r:', val_macro_r)
        print('val_micro_f1:', val_micro_f1)
        print('val_macro_f1:', val_macro_f1)
        print('val_micro_auc_roc:', val_micro_auc_roc)
        print('val_macro_auc_roc:', val_macro_auc_roc)

        if val_micro_f1 > best_val_micro_f1:
            best_val_micro_f1 = val_micro_f1

            #保存模型
            torch.save(model,'model_%d.pkl'%labelNum)
        early_stopping(val_micro_f1, model)
        if early_stopping.early_stop:
            print('early stopping')
            break

    #测试模型
    test_model()



