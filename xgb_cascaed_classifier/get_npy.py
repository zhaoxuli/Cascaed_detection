# -*- coding: utf-8 -*-
import os
#import xgboost as xgb
import cv2
import numpy as np
import random
from time import clock as now


illegal_txt = './illegal_img.txt'


def get_binary(size,data_ctx,data_type,illegal_ctx):
    print len(data_ctx)
    lines = map(lambda x:x.strip(),data_ctx)
    ctx = map(lambda x:x.split(' '),lines)
    random.shuffle(ctx)
    unit = len(ctx)/100
    print 'all_image:',len(ctx)
    count = 0
    unit_mat =np.zeros(size[0]*size[1]+1)
    log = 0
    for img_url, truth in ctx:
        if count ==0:
            time1 = now()
        src = cv2.imread(img_url,0)
        try:
            dst_ary = src.reshape(size[0]*size[1])
            label = np.array([int(truth)])
            in_ary = np.hstack((dst_ary,label))
            unit_mat = np.row_stack((unit_mat,in_ary))
        except:
            ctx = img_url+' '+data_type+'\n'
            illegal_ctx.write(ctx)
        if count == unit:
            print '1% count is ',count
            print unit_mat.shape
            unit_mat =unit_mat[1:,:]
            np.save(data_type+'_'+str(log)+'.npy',unit_mat)
            time2 =now()
            count =0
            log = log+1
            print 'cost time:',time2-time1
            print 'done ',log,'%'
            print unit_mat.shape
            unit_mat =np.zeros(size[0]*size[1]+1)
        count = count+1

if  __name__ =='__main__':
    train_txt = './train_anno.txt'
    test_txt = './test_anno.txt'

    illegal_ctx = open(illegal_txt,'w')
    train_ctx =open(train_txt).readlines()
    test_ctx =open(test_txt).readlines()
    get_binary([60,40],train_ctx,'train',illegal_ctx)
    get_binary([60,40],test_ctx,'test',illegal_ctx)
    illegal_ctx.close()



