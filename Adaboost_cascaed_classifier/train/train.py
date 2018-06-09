#coding=utf-8
from sklearn.metrics import zero_one_loss
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle as  pkl
import time
import numpy as np
import os
import get_best_score  as   gbs

def get_train_matix(data_path):
    mat1 = np.load(data_path+os.sep+'train_0.npy')
    lst = os.listdir(data_path)
    if len(lst) >1:
        for i in range(1,len(lst)):
            mat_add = np.load(data_path+os.sep+'train_'+str(i)+'.npy')
            mat1 = np.concatenate((mat1,mat_add),axis=0)
            np.random.shuffle(mat1)
            #print ('finished '+str(i)+'%')
            mat_sort = mat1[:,0:-1]
            label_sort = mat1[:,-1]
    else:
        np.random.shuffle(mat1)
        #print ('finished '+str(i)+'%')
        mat_sort = mat1[:,0:-1]
        label_sort = mat1[:,-1]
    return mat_sort,label_sort,mat1

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
        train_mat,train_label,data_mat = get_train_matix(data_path)
    else:
        test_mat ,test_label,data_mat= get_train_matix(data_path)
        train_mat,train_label,data_mat = get_train_matix(data_path)
    unit_count = str(unit_count)
    now = time.time()
    iter_count = 0
    print  'Training Adaboost unit ',unit_count,'...'
    bdt = AdaBoostClassifier(DecisionTreeClassifier(criterion=params['dt_criterion'], splitter=params['dt_spliter'],
                                                      max_depth=params['dt_max_depth'],
                                                      min_samples_split=params['dt_min_samples_split'],
                                                      min_samples_leaf=params['dt_min_samples_leaf'],
                                                      class_weight=params['dt_class_weight'],
                                                      min_weight_fraction_leaf=0., max_features=None,
                                                      random_state=None, max_leaf_nodes=None,min_impurity_decrease=0.,
                                                      min_impurity_split=None,  presort=False),
                             algorithm=params['algorithm'],
                             learning_rate =params["learning_rate"],
                             n_estimators=params["n_estimators"])
    bdt.fit(train_mat,train_label)
    #save model
    file = open(model_path+os.sep+'/ada_uint'+str(unit_count)+'.pkl','w')
    pkl.dump(bdt,file)
    #out log
    for y_pred in bdt.staged_predict(train_mat):
        loss = zero_one_loss(y_pred,train_label)
        print  'Iter_count [',iter_count,'] loss is :',loss
        iter_count +=1
    out_lst = bdt.predict_proba(train_mat)
    #save  pre_lst
    recorder_path ='unit_'+str(unit_count)+ '_record.txt'
    print 'train_out/label len:',len(out_lst),len(train_label)
    pre_lst = []
    label_lst = []
    out_f = open(recorder_path,'w')
    for ele in out_lst:
        #1 score
        pre_lst.append(ele[1])
        out_f.write(str(ele[1])+' ')
    out_f.write('\n')
    for ele in train_label:
        #truth
        label_lst.append(ele)
        out_f.write(str(ele)+' ')
    out_f.close
    #  get_two_lst
    print len(pre_lst),len(label_lst)
    cost_time = time.time()-now
    print "end ......",'\n',"cost time:",cost_time,"(s)......"
    return  data_mat,pre_lst,label_lst

if __name__ =='__main__':
    Ada_params = {
        'dt_criterion':'gini',
        'dt_spliter':'random',
        'dt_max_depth':2,
        'dt_min_samples_split':10,
        'dt_max_features':'log2,',
        'dt_min_samples_leaf':5,
        'dt_max_leaf_nodes':None,
        'dt_class_weight':'balanced',

        'n_estimators':100,
        'learning_rate':0.7,
        'algorithm':'SAMME.R',
        'random_state':None
        }
    data_path ='../feature_data_npy/haar_feature'
    unit_count = 0
    model_path = './'
    train(data_path,Ada_params,unit_count,model_path)
