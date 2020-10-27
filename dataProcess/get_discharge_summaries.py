# -*- coding:utf-8 -*-
"""
@Time: 2019/06/16 17:04
@Author: Shanshan Wang
@Version: Python 3.7
@Function:从NOTEEVENTS 文件中提取到需要用到信息，最主要的是discharge summaries
"""
import csv
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm
import os

# retain only alphanumeric
tokenizer=RegexpTokenizer(r'\w+')

def write_discharge_summaries(out_file):
    #notes_file=os.path.join(MIMIC_III_DIR,'NOTEEVENTS.csv')
    notes_file='F:\数据资源\MIMIC\MIMIC-2\\NOTEEVENTS.csv\\NOTEEVENTS.csv'
    print('Processing notes files....')
    with open(notes_file, 'r') as csvfile:
        notereader = csv.reader(csvfile)
        # header
        next(notereader)

        with open(out_file, 'w',newline='') as outfile:
            print("writing to %s" % (out_file))
            writer=csv.writer(outfile)
            writer.writerow(['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'TEXT'])

            i = 0
            for line in tqdm(notereader):
                # subj = int(line[1])
                category = line[6]
                if category == "Discharge summary":
                    note = line[10]

                    # 只保留主诉数据和现病史数据
                    chief = ''
                    index_1 = note.find('CHIEF COMPLAINT:')
                    index_2 = note.find('Chief Complaint:')
                    if index_1 != -1 or index_2 != -1:
                        if index_1 != -1:
                            index = index_1
                        else:
                            index = index_2
                        note_ = note[index + 16:]
                        # print('note_:',note_)
                        # 寻找下一个分号的位置 中间的字符串就是主诉数据
                        fenhao = note_.find(':')
                        if fenhao != -1:
                            chief = note_[:fenhao]
                            # print('chief:',chief)
                    # 同理 寻找现病史数据
                    hist = ''
                    index_1 = note.find('History of Present Illness:')
                    index_2 = note.find('HISTORY OF PRESENT ILLNESS:')
                    index_3 = note.find('HISTORY OF THE PRESENT ILLNESS:')
                    index_4 = note.find('PRESENT ILLNESS:')
                    if index_1 != -1 or index_2 != -1 or index_3 != -1 or index_4 != -1:
                        if index_1 != -1:
                            index = index_1
                        elif index_2 != -1:
                            index = index_2
                        elif index_3 != -1:
                            index = index_3
                        else:
                            index = index_4
                        note_ = note[index + 27:]
                        # print('note_:',note_)
                        # 寻找下一个分号的位置 中间的字符串就是主诉数据
                        fenhao = note_.find(':')
                        if fenhao != -1:
                            hist = note_[:fenhao]

                            # print('hist:',hist)
                    # 有时候主诉数据与现病史数据在一块
                    index_1 = note.find('CHIEF COMPLAINT/HISTORY OF PRESENT ILLNESS:')

                    index_2 = note.find('Chief Complaint/History of Present Illness:')
                    if index_1 != -1 or index_2 != -1:
                        if index_1 != -1:
                            index = index_1
                        else:
                            index = index_2
                        note_ = note[index + 43:]
                        # print('note_:',note_)
                        # 寻找下一个分号的位置 中间的字符串就是主诉数据
                        fenhao = note_.find(':')
                        if fenhao != -1:
                            print(',,,,,,,,,,,,,,,,,,,,,,,,')
                            note = note_[:fenhao]

                    print('==================')
                    if len(chief + hist) != 0:
                        note = chief + hist
                        print('***************')
                    else:
                        print(note)
                    # print(note)
                    index = note.find('Service:')
                    # print(index)
                    if index != -1:
                        note = note[index:]
                    index = note.find('Discharge Instructions')
                    if index != -1:
                        note = note[:index]
                    index = note.find('Discharge Condition')
                    if index != -1:
                        note = note[:index]
                    index = note.find('first name')
                    if index != -1:
                        note = note[:index]

                        #print(note)
                    # tokenize, lowercase and remove numerics
                    tokens = [t.lower() for t in tokenizer.tokenize(note) if not t.isnumeric()]
                    text =' '.join(tokens).strip()
                    writer.writerow([line[1], line[2], line[4], text])
                i += 1
    return out_file