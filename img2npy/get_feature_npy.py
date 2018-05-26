# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import random
import haar_like as haar
import lbp_like as lbp
from time import clock as now


illegal_txt = './illegal_img.txt'


def get_binary(size,data_ctx,data_type,illegal_ctx,lbp_dir,haar_dir):
    print len(data_ctx)
    lines = map(lambda x:x.strip(),data_ctx)
    ctx = map(lambda x:x.split(' '),lines)
    random.shuffle(ctx)
    unit = len(ctx)/100
    print 'all_image:',len(ctx)
    count = 0
    #'uniform' style is 10*4*4 dims  +1 truth
    lbp_unit_mat =np.zeros(161)
    # haar.haar_filter(Iteral,12,8,'edge_y',step =2) out.shape is 600
    haar_unit_mat =np.zeros(601)
    log = 0
    for img_url, truth in ctx:
        if count ==0:
            time1 = now()
        src = cv2.imread(img_url,0)
        try:
            lbp_dst_ary = lbp.lbp_like(src)
            Iteral = haar.Integral_feature(src)
            haar_dst_ary = haar.haar_filter(Iteral,12,8,'edge_y',step =2)
            label = np.array([int(truth)])
            lbp_in_ary = np.hstack((lbp_dst_ary,label))
            haar_in_ary = np.hstack((haar_dst_ary,label))
            lbp_unit_mat = np.row_stack((lbp_unit_mat, lbp_in_ary))
            haar_unit_mat = np.row_stack((haar_unit_mat,haar_in_ary))
        except:
            ctx = img_url+' '+data_type+'\n'
            illegal_ctx.write(ctx)
        if count == unit:
            print '1% count is ',count
            lbp_unit_mat =lbp_unit_mat[1:,:]
            haar_unit_mat =haar_unit_mat[1:,:]
            np.save(haar_dir+os.sep+data_type+'_'+str(log)+'.npy',haar_unit_mat)
            np.save(lbp_dir+os.sep+data_type+'_'+str(log)+'.npy',lbp_unit_mat)
            time2 =now()
            count =0
            log = log+1
            print 'cost time:',time2-time1
            print 'done ',log,'%'
            print lbp_unit_mat.shape
            print haar_unit_mat.shape
            #'uniform' style is 10*4*4 dims  +1 truth
            lbp_unit_mat =np.zeros(161)
            # haar.haar_filter(Iteral,12,8,'edge_y',step =2) out.shape is 600
            haar_unit_mat =np.zeros(601)

        count = count+1

if  __name__ =='__main__':
    train_txt = './train_anno.txt'
    test_txt = './test_anno.txt'

    illegal_ctx = open(illegal_txt,'w')
    train_ctx =open(train_txt).readlines()
    test_ctx =open(test_txt).readlines()

    haar_dir = './haar_feature'
    lbp_dir = './lbp_feature'
    if os.path.exists(haar_dir) ==False:
        os.makedirs(haar_dir)
    if os.path.exists(lbp_dir) ==False:
        os.makedirs(lbp_dir)
    get_binary([60,40],train_ctx,'train',illegal_ctx,lbp_dir,haar_dir)
    get_binary([60,40],test_ctx,'test',illegal_ctx,lbp_dir,haar_dir)

    illegal_ctx.close()



