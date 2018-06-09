import numpy as np
from sklearn.metrics import zero_one_loss
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle as pkl
import os

def read_from_txt(txt_path):
    ctx = open(txt_path,'r').readlines()
    lst1 =map(float,ctx[0].strip().split(' '))
    lst2 =map(float,ctx[1].strip().split(' '))
    return  lst1,lst2

def predict(mod_path,data_path):
    x = np.load(data_path)
    label_lst = x[:,-1]
    x = x[:,:-1]
    f = open(mod_path,'r')
    mod = pkl.load(f)
    pre_lst = mod.predict_proba(x)
    obj_score_lst = []
    for   ele in pre_lst:
        obj_score_lst.append(float(ele[1]))
    return  obj_score_lst,label_lst

def cacul(label_lst,pred_lst,thres):
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

def get_best_thres(target,thres_back,thres,pre_lst,label_lst,count):
    steps = float(abs(thres_back-thres)/2)
    recall,fppw = cacul(label_lst,pre_lst,thres)
    count =count+1
    if count >30:
        return thres,fppw,recall
    if recall < target:
        thres_new = thres - steps
        return  get_best_thres(target,thres,thres_new,pre_lst,label_lst,count)
    if recall >=target:
        if (recall - target) >=0.001:
            thres_new = thres +steps
            return   get_best_thres(target,thres,thres_new,pre_lst,label_lst,count)
        else:
            return thres,fppw,recall

def get_next_train_data(thres,pre_lst,label_lst,train_mat,data_path,unit_count):
    print train_mat.shape
    out_mat = np.zeros((train_mat.shape[1]))
    for i  in range(len(pre_lst)):
        if  (pre_lst[i] > thres) or (pre_lst[i]<=thres and  label_lst[i]==1):
            out_mat = np.row_stack((out_mat,train_mat[i]))
    next_data_path = data_path+os.sep+'train_0.npy'
    np.save(next_data_path,out_mat[1:])


def gbs_run(pre_lst,label_lst,recall_target,info_recoder,unit_count,train_mat,next_data_folder):
    print  'gbs_run_pre/label_len:',len(pre_lst),len(label_lst)
    thres,fppw,recall = get_best_thres(recall_target,0,0.99,pre_lst,label_lst,0)
    print 'Finished '+str(unit_count)+' unit train:'
    print '[thres,fppw,recall]',thres,fppw,recall
    f = open(info_recoder,'w')
    str_in = str(unit_count)+' unit train:[thres,fppw,recall]'+str([thres,fppw,recall])+'\n'
    f.write(str_in)
    f.close
    get_next_train_data(thres,pre_lst,label_lst,train_mat,next_data_folder,unit_count)
    return  thres,fppw,recall


if __name__ == '__main__':
    mod_path = './ada_model.pkl'
    data_path = './test_0.npy'
    train_mat = np.load(data_path)
    pre_lst,label_lst = predict(mod_path,data_path)
    #quit()
    recall_target=0.99
    info_recoder = './test_log.txt'
    unit_count= 0
    next_data_path = './'
    a,b,c= gbs_run(pre_lst,label_lst,recall_target,info_recoder,unit_count,train_mat,next_data_path)
