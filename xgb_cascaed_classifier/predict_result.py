import xgboost  as  xgb
import numpy as np
import os

def predict(data_path,unit_count):
    unit_count = str(unit_count)
    model_path = './model/xgb_uint'+unit_count+'.model'
    out_file = open('unit'+unit_count+'_result.txt','w')
    out_ctx = []
    '''laod_model'''
    bst = xgb.Booster({'nthread':4})
    bst.load_model(model_path)
    bst.__hash__

    files = os.listdir(data_path)
    print 'predict',
    for ele in files:
        npy_path = data_path+os.sep+ele
        print '.',
        mat1 = np.load(npy_path)
        dst_ary = mat1[:,:-1]
        label= mat1[:,-1]
        dtest = xgb.DMatrix(dst_ary)
        pred = bst.predict(dtest)
        str_label='|'
        for i in label:
            str_label =str_label+str(i)+' '
        str_pred = '|'
        for i in pred:
            str_pred =str_pred+str(i)+' '
        ctx = ele+str_label +str_pred+'\n'
        out_file.write(ctx)
        out_ctx.append(ctx)
    out_file.close()
    return out_ctx

