# -*- coding: utf-8 -*-
# file: preprocess.py
# author: jackie
# Copyright (C) 2021. All Rights Reserved.

import os
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--inpath", type=str, required=True,default='./raw_data/data1.csv')
parser.add_argument("--folder_name",type=str,required=False,default='./custom')
parser.add_argument("--task",type=str,required=False,default='aptepc')
args = parser.parse_args()

def convert(text, labels):
    # convert label to list
    labels = eval(labels)
    tags = ['O'] * len(text)
    sentiment = ['-999'] * len(text)

    for j in range(len(labels)):
        label = labels[j]
        sentiment_key = labels[j][3]

        if sentiment_key == '正':
            sentiment_value = 'Positive'
        else:
            sentiment_value = 'Negative'

        tags[label[4][0]] = 'B-ASP'
        sentiment[label[4][0]] = sentiment_value

        k = label[4][0] + 1
        while k < label[4][1]:
            tags[k] = 'I-ASP'
            sentiment[k] = sentiment_value

            k += 1

    return text, tags, sentiment


def convert_sentiment(sentiment_key):
    if sentiment_key == '正':
        sentiment_value = 'Positive'
    else:
        sentiment_value = 'Negative'
    return sentiment_value


def convert_apc(text, label):
    label_update = [(i[0], i[3], i[4]) for i in eval(label)]
    label_update = list(set(label_update))

    str1_list = []
    str2_list = []
    str3_list = []

    for j in range(len(label_update)):
        str1 = text[:label_update[j][2][0]] + '$T$ ' + text[label_update[j][2][1]:]
        str1_list.append(str1)
        str2_list.append(label_update[j][0])
        str3_list.append(convert_sentiment(label_update[j][1]))

    return str1_list, str2_list, str3_list

def convert_to_atepc(inpath, dist_fname, flag):
    # 写之前，先检验文件是否存在，存在就删掉
    if os.path.exists(dist_fname):
        os.remove(dist_fname)
    f1 = open(dist_fname, 'w', encoding='utf8')

    data = pd.read_csv(inpath)
    # train test split
    x_train, x_test = train_test_split(data, test_size=0.2, random_state=42)

    if flag == 'train':
        data_res = x_train.iloc[:,:].reset_index()
    else:
        data_res = x_test.iloc[:,:].reset_index()
    #print (data_res.head())
    for i in range(len(data_res)):
        text, label = data_res['text'][i], data_res['tag_sentiment_list'][i]
        text, tags, sentiment = convert(text, label)

        for word, tag, sen in zip(text, tags, sentiment):
            if word not in ['，', '。', ' ', '\xa0', '\u2006','\u3000','\u2002','\u2003','\u2005','\x0c','\u2028','\u2009','\u200a']:
                f1.write(word + ' ' + tag + ' ' + sen + '\n')
            else:
                f1.write("\n")

        f1.write("\n")
    f1.close()
    print ("process atepc finished!")

def convert_to_apc(inpath, dist_fname, flag):
    # 写之前，先检验文件是否存在，存在就删掉
    if os.path.exists(dist_fname):
        os.remove(dist_fname)
    f1 = open(dist_fname, 'w', encoding='utf8')

    data = pd.read_csv(inpath)
    # train test split
    x_train, x_test = train_test_split(data, test_size=0.2, random_state=42)

    if flag == 'train':
        data_res = x_train.iloc[:,:].reset_index()
    else:
        data_res = x_test.iloc[:,:].reset_index()
    #print (data_res.head())
    for i in range(len(data_res)):
        text, label = data_res['text'][i], data_res['tag_sentiment_list'][i]
        str1_list, str2_list, str3_list = convert_apc(text, label)

        for x1, x2, x3 in zip(str1_list, str2_list, str3_list):
            f1.write(x1 + '\n')
            f1.write(x2 + '\n')
            f1.write(x3 + '\n')

    f1.write("\n")
    f1.close()

    print ("process apc finished!")

def main(inpath, folder_name, task):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    if task == 'aptepc':
        # get folder name
        folder_name_prefix = folder_name.split('/')[-1]
        dist_train_fname = os.path.join(folder_name_prefix,folder_name_prefix+'.train.txt.atepc')
        dist_test_fname = os.path.join(folder_name_prefix,folder_name_prefix+'.test.txt.atepc')
        #process train
        convert_to_atepc(inpath, dist_train_fname, 'train')
        print ("<<< finish training data preprocess")
        #process test
        convert_to_atepc(inpath, dist_test_fname, 'test')
        print ("<<< finish test data preprocess")
    elif task == 'apc':
        # get folder name
        folder_name_prefix = folder_name.split('/')[-1]
        dist_train_fname = os.path.join(folder_name_prefix, folder_name_prefix + '.train.txt')
        dist_test_fname = os.path.join(folder_name_prefix, folder_name_prefix + '.test.txt')
        # process train
        convert_to_apc(inpath, dist_train_fname, 'train')
        print ("<<< finish training data preprocess")
        # process test
        convert_to_apc(inpath, dist_test_fname, 'test')
        print ("<<< finish test data preprocess")


main(args.inpath, args.folder_name,args.task)