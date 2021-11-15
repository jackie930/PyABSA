# -*- coding: utf-8 -*-
# file: metrics.py
# author: jackie
# Copyright (C) 2021. All Rights Reserved.

import os
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from pyabsa import ATEPCCheckpointManager

parser = argparse.ArgumentParser()
parser.add_argument("--checkppoint", type=str, required=False,default='checkpoints/lcf_atepc_cdw_apcacc_96.12_apcf1_91.88_atef1_96.85')
parser.add_argument("--data_path",type=str,required=False,default='raw_data/data1.csv')
args = parser.parse_args()


def infer(aspect_extractor,examples):
    atepc_result = aspect_extractor.extract_aspect(inference_source=examples,
                                                   save_result=True,
                                                   print_result=True,  # print the result
                                                   pred_sentiment=True,  # Predict the sentiment of extracted aspect terms
                                                   )
    return atepc_result

def init(checkppoint):
    #init modelimport pyabsa
    aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint=checkppoint,
                                                                   auto_device=True  # False means load model on CPU
                                                                   )
    return aspect_extractor


def post_process_label(label_list):
    x = list(set([(i[0], i[3]) for i in eval(label_list)]))
    return x


def post_process_prediction(x):
    # replace sentiment
    repalce_dict = {'Positive': '正', 'Negative': '负'}
    sentiment = [repalce_dict[i] for i in x['sentiment']]
    res = [(x['aspect'][i].replace(" ", ""), sentiment[i]) for i in range(len(sentiment))]
    return res


def calculate_score_total(prediction,labels):
    n_tp, n_gold, n_pred = 0, 0, 0

    for i in range(len(prediction)):
        gold_pt = labels[i]
        # print (gold_pt)
        pred_pt = prediction[i]
        # print (pred_pt)
        n_gold += len(gold_pt)
        n_pred += len(pred_pt)
        # print (n_gold)
        # print (n_pred)

        for t in pred_pt:
            if t in gold_pt:
                n_tp += 1

        precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
        recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
        scores = {'precision': precision, 'recall': recall, 'f1': f1}
    return scores

def get_label_term_list(labels):
    res = []
    for i in range(len(labels)):
        res.append([j[0] for j in labels[i]])
    return res


def calculate_score_ate(prediction, labels):
    n_tp, n_gold, n_pred = 0, 0, 0

    labels_term = get_label_term_list(labels)
    for i in range(len(prediction)):

        gold_pt = labels_term[i]
        # print (gold_pt)
        pred_pt = prediction[i]
        # print (pred_pt)
        n_gold += len(gold_pt)
        n_pred += len(pred_pt)
        # print (n_gold)
        # print (n_pred)

        for t in pred_pt:
            if t[0] in gold_pt:
                n_tp += 1

        precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
        recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
        scores = {'precision': precision, 'recall': recall, 'f1': f1}
    return scores


def find_sentiment(x):
    res = {}
    for i in range(len(x)):
        # print (x[i])
        res[x[i][0]] = x[i][1]
    return res


def calculate_score_apc(prediction, labels):
    # 只取aspect提取正确的部分,计算
    n_tp, n_gold, n_pred = 0, 0, 0
    labels_term = get_label_term_list(labels)

    for i in range(len(prediction)):
        label_dict = find_sentiment(labels[i])

        for t in prediction[i]:
            # print (t)
            # print (labels_term[i])
            if t[0] in labels_term[i]:
                gold_pt = label_dict[t[0]]
                pred_pt = t[1]
                # print (pred_pt)
                n_gold += 1
                n_pred += 1
                if gold_pt == pred_pt:
                    n_tp += 1

        precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
        recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
        scores = {'precision': precision, 'recall': recall, 'f1': f1}

    return scores

def main(data_path,checkppoint):
    aspect_extractor = init(checkppoint)
    # label list
    data = pd.read_csv(data_path)
    x_train, x_test = train_test_split(data, test_size=0.2, random_state=42)
    to_predict = x_test['text'].tolist()

    prediction = [post_process_prediction(i) for i in infer(aspect_extractor,to_predict)]
    # prediction list
    labels = [post_process_label(i) for i in x_test['tag_sentiment_list'].tolist()]

    ## sample prediction: [[('性价比', '正')], [('早餐', '正'), ('不错', '正')]]
    ## sample labels: [[('性价比', '正')], [('不错', '正'), ('早餐', '正')]]

    score = calculate_score_total(prediction,labels)
    print ("end to end result: ", score)
    score2 = calculate_score_ate(prediction, labels)
    print ("term extraction result: ", score2)
    score3 = calculate_score_apc(prediction, labels)
    print ("sentiment classification result: ", score3)

main(args.data_path, args.checkppoint)