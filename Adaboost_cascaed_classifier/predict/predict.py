# -*- coding: utf-8 -*-
import numpy as np
import os
from sklearn.metrics import zero_one_loss
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle as pkl


def cacul(label_lst,pred_lst,thres=1):
    TP,FP,FN,TN = [0,0,0,0]
    nums = len(label_lst)
    for i in range(0,nums):
        if (float(pred_lst[i]) >=thres) and (float(label_lst[i])==1):
            TP = TP+1
        if (float(pred_lst[i]) >=thres) and (float(label_lst[i])==0):
            FP = FP+1
        if (float(pred_lst[i]) < thres) and (float(label_lst[i])==1):
            FN = FN+1
        if (float(pred_lst[i]) < thres) and (float(label_lst[i])==0):
            TN = TN+1
    Recall = float(TP)/(float(TP)+float(FN)+0.0000001)
    FPPW = float(FP)/(float(FP)+float(FN)+0.0000001)
    print 'reacall:' ,Recall,
    print ' FPPW:',FPPW,
    print ' thres ',thres
    print '______________'
    return  Recall,FPPW


def  eval_test(test_path,model_path,thres_lst):
    #load data
    lst = os.listdir(test_path)

    test_data_path= test_path+os.sep+lst[0]
    mat = np.load(test_data_path)
    data_mat = mat[:,:-1]
    label_mat = mat[:,-1]
    if  len(lst) >=1:
        for i in range(1,len(lst)):
            ele = lst[i]
            test_data_path = test_path+os.sep+ele
            x = np.load(test_data_path)
            label_lst = x[:,-1]
            x = x[:,:-1]
            data_mat = np.row_stack((data_mat,x))
            #print data_mat.shape
            #print  label_mat.shape
            #print  label_lst.shape
            label_mat = np.concatenate((label_mat,label_lst))
    label_lst = label_mat
    print 'all_data_num:',len(data_mat)
    models = os.listdir(model_path)
    models_lst = map(lambda x:model_path+os.sep+x,models)
    out_lst = []
    for ele  in models_lst:
        f = open(ele,'r')
        mod = pkl.load(f)
        #predict
        prd_lst = mod.predict_proba(data_mat)
        out_lst.append(prd_lst)
    score_lst = [1]*len(out_lst[0])
    for i in range(len(out_lst)):
        thres = thres_lst[i]
        for  j in range(len(out_lst[i])):
            ele =out_lst[i][j][1]
            if ele < thres:
                score_lst[j] =0
    #print score_lst
    #print label_lst
    cacul(label_lst,score_lst)


if  __name__ == '__main__':
    model_path = '../train/models'
    test_path = '../../data/test_data'
    cascade_info_txt = './casced_setting.txt'
    f = open(cascade_info_txt,'r').readlines()
    unit_num = len(f)
    thres_lst = []
    for ele in f:
        thres_lst.append(float(ele.split('[')[-1].split(',')[0]))
    eval_test(test_path,model_path,thres_lst)
