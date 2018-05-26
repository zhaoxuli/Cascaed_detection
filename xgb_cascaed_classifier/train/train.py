#coding=utf-8
import xgboost as xgb
import time
import numpy as np
import os

def get_train_matix(data_path):
    mat1 = np.load(data_path+os.sep+'train_0.npy')
    lst = os.listdir(data_path)
    for i in range(1,len(lst)):
        mat_add = np.load(data_path+os.sep+'train_'+str(i)+'.npy')
        mat1 = np.concatenate((mat1,mat_add),axis=0)
        #print ('finished '+str(i)+'%')
        mat_sort = mat1[:,0:-1]
        label_sort = mat1[:,-1]
    return mat_sort,label_sort

def get_test_matix(data_path):
    mat1 = np.load(data_path+os.sep+'test_data/test_0.npy')
    for i in range(1,100):
        mat_add = np.load(data_path+os.sep+'test_data/'+'test_'+str(i)+'.npy')
        mat1 = np.concatenate((mat1,mat_add),axis=0)
        #print ('finished '+str(i)+'%')
        mat_sort = mat1[:,0:-1]
        label_sort = mat1[:,-1]
    return mat_sort,label_sort

def  train(data_path,params,unit_count,model_path):
    print 'loding data...'
    if unit_count==0:
        test_mat ,test_label= get_test_matix(data_path)
        data_path = data_path+os.sep+'train_data'
        train_mat,train_label = get_train_matix(data_path)
    else:
        test_mat ,test_label= get_train_matix(data_path)
        train_mat,train_label = get_train_matix(data_path)
    unit_count = str(unit_count)
    now = time.time()
    xgtrain= xgb.DMatrix(train_mat,label=train_label)
    xgval= xgb.DMatrix(test_mat,label=test_label)

    plst = list(params.items())

    num_rounds = 100 # 迭代你次数
    watchlist = [(xgtrain, 'train'),(xgval, 'val')]

    model = xgb.train(plst, xgtrain, num_rounds, watchlist,early_stopping_rounds=100)
    model.save_model(model_path+os.sep+'/xgb_uint'+unit_count+'.model') # 用于存储训练出的模型

    cost_time = time.time()-now
    print "end ......",'\n',"cost time:",cost_time,"(s)......"# -*- coding: utf-8 -*-
