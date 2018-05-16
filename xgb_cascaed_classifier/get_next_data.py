import numpy as np
import os

def cacul(ctx,thres):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for ele in ctx:
        lst = ele.split('|')
        label_lst = lst[1].strip().split(' ')
        pred_lst = lst[2].strip().split(' ')
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


def get_best_thres(target,thres_back,thres,ctx):
    steps = float(abs(thres_back-thres)/2)
    recall,fppw = cacul(ctx,thres)
    if recall < target:
        thres_new = thres - steps
        return  get_best_thres(target,thres,thres_new,ctx)
    if recall >=target:
        if (recall - target) >=0.001:
            thres_new = thres +steps
            return   get_best_thres(target,thres,thres_new,ctx)
        else:
            return thres,fppw,recall

def get_next_train(ctx,thres):
    ctx_in = []
    for ele in ctx:
        lst = ele.split('|')
        npy_key = lst[0]
        if npy_key[0:4] != 'test':
            label_lst = lst[1].strip().split(' ')
            pred_lst = lst[2].strip().split(' ')
            nums = len(label_lst)
            str_save = npy_key+' '
            for i in range(0,nums):
                if (float(pred_lst[i]) >=thres) or ((float(pred_lst[i]) < thres) and (float(label_lst[i])==1)):
                    str_save = str_save+ str(i)+' '
            str_save = str_save + '\n'
            ctx_in.append(str_save)
    return ctx_in

def save_np(out_path,in_ary,count,mat_new):
    mat_new = np.row_stack((mat_new,in_ary))
    if count % 899 == 0 and count != 0:
        mat_new= mat_new[1:,:]
        np.save(out_path+os.sep+'train'+'_'+str((count//899)-1)+'.npy',mat_new)
        print '.',
    return count+1,mat_new


def run(ctx,data_path,recall_target,out_path):
    if os.path.exists(out_path) == False:
        os.makedirs(out_path)
    thres,fppw,recall = get_best_thres(recall_target,0,0.99,ctx)
    next_ctx = get_next_train(ctx,thres)
    count = 0
    print  'saving next train samples...',

    for ele in next_ctx:
        ele = ele.strip()
        lst = ele.split(' ')
        npy_key = lst[0]
        #print npy_key
        ID_lst = lst[1:]
        np_url = data_path+os.sep+npy_key
        mat = np.load(np_url)
        h,w = mat.shape
        for  ID  in ID_lst:
            if count  ==0:
                mat_new =np.zeros(w)
            in_ary = mat[int(ID)]
            count,mat_new = save_np(out_path,in_ary,count,mat_new)
            if count %900 ==0:
                mat_new =np.zeros(w)
    np.save(out_path+os.sep+'train'+'_'+str((count//899))+'.npy',mat_new)
    return  thres,fppw,recall

if __name__=='__main__':
    ctx = open('./unit0_result.txt','r').readlines()
    data_path ='../unit_1/data_all//train_data/'
    recall_target=0.99
    out_path='./uinit_0_data'
    run(ctx,data_path,recall_target,out_path)
