# -*- coding:utf-8 -*-
"""
@Time: 2019/06/16 17:05
@Author: Shanshan Wang
@Version: Python 3.7
@Function:Pre-train embeddings using gensim w2v implementation (CBOW by default)
"""
import gensim.models.word2vec as w2v
import csv

processed_data_dir = r'F:\MYPAPERS\GraphMatch\code\data'
class ProcessedIter(object):
    def __init__(self,Y,filename):
        self.filename=filename
    def __iter__(self):
        with open(self.filename) as f:
            r=csv.reader(f)
            next(r)
            for row in r:
                yield (row[3].split())

def word_embeddings(Y,notes_file,embedding_size,min_count,n_iter):
    modelname='Processed_%s.w2v'%(Y)
    sentences=ProcessedIter(Y,notes_file)

    model=w2v.Word2Vec(size=embedding_size,min_count=min_count,workers=4,iter=n_iter)
    print('building word2vec vocab on %s....'%(notes_file))

    model.build_vocab(sentences)
    print('training......')
    model.train(sentences,total_examples=model.corpus_count,epochs=model.iter)
    out_file='%s/%s'%(processed_data_dir,modelname)
    model.save(out_file)
    return out_file

import os
import gensim.models
from tqdm import tqdm
import numpy as np

def gensim_to_emebddings(wv_file,vocab_file,Y,outfile=None):
    model=gensim.models.Word2Vec.load(wv_file)
    wv=model.wv
    #free up memory
    del model

    vocab=set()
    with open(vocab_file,'r') as vocabfile:
        for i,line in enumerate(vocabfile):
            line=line.strip()
            if line!='':
                vocab.add(line)
    ind2w={i+1:w for i,w in enumerate(sorted(vocab))}

    W,words=build_matrix(ind2w,wv)
    if outfile is None:
        outfile=wv_file.replace('.w2v','.embed')
    #smash that save button
    save_embeddings(W,words,outfile)

def build_matrix(ind2w,wv):
    '''
    Go through vocab in order. Find vocab word in wv.index2word, then call wv.word_vec(wv.index2word[i]).
        Put results into one big matrix.
        Note: ind2w starts at 1 (saving 0 for the pad character), but gensim word vectors starts at 0
    :param ind2w:
    :param wv:
    :return:
    '''
    W=np.zeros((len(ind2w)+1,len(wv.word_vec(wv.index2word[0]))))
    words=['PAD']
    W[0][:]=np.zeros(len(wv.word_vec(wv.index2word[0])))
    for idx,word in tqdm(ind2w.items()):
        if idx>=W.shape[0]:
            break
        W[idx][:]=wv.word_vec(word)
        words.append(word)
    return W,words

def save_embeddings(W,words,outfile):
    with open(outfile,'w') as o:
        #pad token already included
        for i in range(len(words)):
            line=[words[i]]
            line.extend([str(d) for d in W[i]])
            o.write(' '.join(line)+'\n')

