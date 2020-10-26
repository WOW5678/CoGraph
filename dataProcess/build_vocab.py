# -*- coding:utf-8 -*-
"""
@Time: 2019/06/16 17:05
@Author: Shanshan Wang
@Version: Python 3.7
@Function:This script reads a sorted training dataset and builds a vocabulary of terms of given size
    Output: txt file with vocab words
    Drops any token not appearing in at least vocab_min notes
    This script could probably be replaced by using sklearn's CountVectorizer to build a vocab
    要完成的功能就是只保留那些出现在至少3个文档中的单词（写入文件），那些没有出现在3个文档中的单词将会被移除
"""
import csv
import numpy as np
import operator

from collections import defaultdict
from scipy.sparse  import csr_matrix
from collections import Counter

def build_vocab(vocab_min,infile,vocab_filename):
    '''

    :param vocab_min:how many documents a word must appear in to be kept
    :param infile:(training) data file to build vocabulary from
    :param vocab_filename:name for the file to output
    :return:
    '''
    with open(infile,'r') as csvfile:
        reader=csv.reader(csvfile)
        next(reader)
        dict_final={} #key为单词，vlaue为每个单词出现在多少个文档中
        print('Reading in data......')

        for row in reader:
            dict={}
            #row[2]表示的是文本
            for item in row[2].split():
                if item not in dict:
                    dict[item]=1
            #更新最终的dict_final
            for key,value in dict.items():
                if key not in dict_final:
                    dict_final[key] = dict[key]
                else:
                    dict_final[key] = dict_final[key] + dict[key]
        #打印dict_final
        print(dict_final)
        # 保存出现次数>=3的单词 并写入文档中
        print('Writing output')
        with open(vocab_filename,'w') as vocab_file:
            for word in dict_final:
                if dict_final[word]>=3:
                    vocab_file.write(word+'\n')





