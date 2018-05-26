import xgboost  as  xgb
import numpy as np
import os

def predict(model_path,dtest):
    bst = xgb.Booster({'nthread':4})
    bst.load_model(model_path)
    bst.__hash__
    pred = bst.predict(dtest)
    return pred

def cacul(ctx,thres):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for ele in ctx:
        lst = ele.split('|')
        #npy_key = lst[0]
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
    print 'reacall:' ,Recall
    print 'FPPW:',FPPW
    print 'thres ',thres
    print '______________'
    return  Recall


if __name__  =='__main__':
    data_lst = ['../unit_classifier/unit_1/data_all/train_data','../unit_classifier/unit_1/data_all/test_data']
    #data_lst = ['../unit_classifier/unit_1/data_all/test_data']
    #ref_size = [ref_h,ref_w]
    ref_h = 60
    ref_w = 40
    thres_lst = [0.37125,0.564609375,0.68642578125,0.69196975708 ,0.692075500488 ]
    models_nums = len(thres_lst)
    model_path = '../unit_classifier/unit_casced/model'
    model_lst = os.listdir(model_path)
    out_file = open('finnal_result.txt','w')

    for data_path in data_lst:
        files = os.listdir(data_path)
        for ele in files:
            npy_path = data_path+os.sep+ele
            print npy_path
            mat1 = np.load(npy_path)
            dst_ary = mat1[:,:-1]
            label= mat1[:,-1]
            str_label='|'
            str_pred = '|'
            for count in range(len(label)):
                ary_in  =dst_ary[count]
                ary_in  =ary_in.reshape((1,int(ref_h)*int(ref_w)))
                dtest = xgb.DMatrix(ary_in)
                for i in range(0,models_nums):
                    model_url = model_path+os.sep+model_lst[i]
                    pred = predict(model_url,dtest)
                    if  pred < thres_lst[i]:
                        pred = 0
                        break
                    else:
                        pred = 1
                str_label =str_label+str(label[count])+' '
                str_pred =str_pred+str(pred)+' '

            ctx = ele+str_label +str_pred+'\n'
            out_file.write(ctx)
    out_file.close()
    out_ctx = open('finnal_result.txt','r').readlines()
    cacul(out_ctx,1)

