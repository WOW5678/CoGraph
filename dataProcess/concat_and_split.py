# -*- coding:utf-8 -*-
"""
@Time: 2019/06/16 17:04
@Author: Shanshan Wang
@Version: Python 3.7
@Function:Concatenate the labels with the notes data and split using the saved splits
"""
import csv
from datetime import datetime
import random
import pandas as pd
from sklearn.model_selection import train_test_split

DATETIME_FORMAT='%Y-%m-%d %H:%M:%S'
processed_data_dir='F:\MYPAPERS\GraphMatch\code\data'

def concat_data(labelsfile,notes_file):
    '''

    :param labelsfile:sorted by hadm id, contains one label per line
    :param notes_file:sorted by hadm id, contains one note per line
    :return:
    '''
    with open(labelsfile,'r') as label_file:
        print('concatenating')
        with open(notes_file,'r') as notes_file_:
            outfilename='%s/note_labeled.csv'%processed_data_dir
            with open(outfilename,'w',newline='') as outfile:
                w=csv.writer(outfile)
                w.writerow(['SUBJECT_ID', 'HADM_ID', 'TEXT', 'LABELS'])

                labels_gen=next_labels(label_file)
                notes_gen=next_notes(notes_file_)

                for i, (subject_id,text,hadm_id) in enumerate(notes_gen):
                    if i%10000==0:
                        print(str(i)+'done')
                    cur_subj,cur_labels,cur_hadm=next(labels_gen)
                    if cur_hadm==hadm_id:
                        w.writerow([subject_id, str(hadm_id), text, ';'.join(cur_labels)])
                    else:
                        print('could not find matching hadm_id. data is probably not sorted correctly')
                        break
    return outfilename

def next_labels(labelsfile):
    '''
    Generator for label sets from the label file
    :param labelsfile:
    :return:
    '''
    labels_reader=csv.reader(labelsfile)
    next(labels_reader) #header

    first_label_line=next(labels_reader)

    cur_subj=first_label_line[0]
    cur_hadm=first_label_line[1]
    cur_labels=[first_label_line[2]]

    for row in labels_reader:
        subj_id=row[0]
        hadm_id=row[1]
        code=row[2]
        #keep reading until you hit a new hadm id
        if hadm_id!=cur_hadm or subj_id!=cur_subj:
            try:
                yield cur_subj,cur_labels,cur_hadm
                cur_labels=[code]
                cur_subj=subj_id
                cur_hadm=hadm_id
            except StopIteration:
                print('Some errors.')
                return
        else:
            #add to the labels and move on
            cur_labels.append(code)
    yield cur_subj,cur_labels,cur_hadm

def next_notes(notesfile):
    '''
     Generator for notes from the notes file
     This will also concatenate discharge summaries and their addenda, which have the same subject and hadm id
    :param notesfile:
    :return:
    '''
    nr=csv.reader(notesfile)
    #header
    next(nr)
    first_note=next(nr)

    cur_subj=first_note[0]
    cur_hadm=first_note[1]
    cur_text=first_note[3]

    for row in nr:
        subj_id=row[0]
        hadm_id=row[1]
        text=row[3]
        #keep reading until you hit a new hadm id
        if hadm_id!=cur_hadm or subj_id!=cur_subj:
            try:
                yield cur_subj,cur_text,cur_hadm
                cur_text=text
                cur_subj=subj_id
                cur_hadm=hadm_id
            except StopIteration:
                print('Some errors.')
                return
        else:
            # concatenate to the discharge summary and move on
            cur_text=' '+text
    yield cur_subj,cur_text,cur_hadm

def split_data(labeledfile,base_name):
    print('SPLITING')
    #对hadm_ids进行随机的分割
    with open(labeledfile,'r') as lf:
        reader=csv.reader(lf)
        next(reader)
        hadm_ids=[row[1] for row in reader]
        train_ids, dev_ids = train_test_split(hadm_ids,
                                            test_size=(1.0/3), random_state=22)
        dev_ids,test_ids=train_test_split(dev_ids,test_size=(1.0/2),random_state=22)
        hadm_ids = {}
        hadm_ids['train']=set(train_ids)
        hadm_ids['dev']=set(dev_ids)
        hadm_ids['test']=set(test_ids)

    #Create and write headers for train,dev,test
    train_name='%s_train_split.csv'%(base_name)
    dev_name='%s_dev_split.csv'%(base_name)
    test_name='%s_test_split.csv'%(base_name)
    train_file = open(train_name, 'w',newline='')
    dev_file = open(dev_name, 'w',newline='')
    test_file = open(test_name, 'w',newline='')

    train_writer=csv.writer(train_file)
    dev_writer=csv.writer(dev_file)
    test_writer=csv.writer(test_file)
    train_writer.writerow(['SUBJECT_ID', 'HADM_ID', 'TEXT', 'LABELS'])
    dev_writer.writerow(['SUBJECT_ID', 'HADM_ID', 'TEXT', 'LABELS'])
    test_writer.writerow(['SUBJECT_ID', 'HADM_ID', 'TEXT', 'LABELS'])

    with open(labeledfile,'r') as lf:
        reader=csv.reader(lf)
        next(reader)
        i=0
        cur_hadm=0
        for row in reader:
            # filter text, write to file according to train/dev/test split
            if i%10000==0:
                print(str(i)+'read')
            hadm_id=row[1]

            if hadm_id in hadm_ids['train']:
                train_writer.writerow(row)
            elif hadm_id in hadm_ids['dev']:
                dev_writer.writerow(row)
            elif hadm_id in hadm_ids['test']:
                test_writer.writerow(row)

            i+=1
        train_file.close()
        dev_file.close()
        test_file.close()
    return train_name,dev_name,test_name


