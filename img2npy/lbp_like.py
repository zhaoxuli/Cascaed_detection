# -*- coding: utf-8 -*-
import math
import numpy as np
import cv2
from skimage.feature import local_binary_pattern as lbp

def lbp_like(img,windows_num=4,P=8,R=1,lbp_type='uniform'):
    LBP_TYPE=('default','ror','uniform','var')
    if lbp_type not in LBP_TYPE:
        return None
    H,W = img.shape
    h_step = H/windows_num
    w_step = W/windows_num
    out_lst = []
    for  start_h in range(0,H-h_step+1,h_step):
        for  start_w in range(0,W-w_step+1,w_step):
            end_h = start_h + h_step
            end_w = start_w + w_step
            rec_img = img[start_h:end_h,start_w:end_w]
            rec_lbp = lbp(rec_img,P,R,lbp_type)
            if lbp_type == 'uniform':
                lst = [0]*(P+2)
            elif lbp_type in ['default','ror']:
                lst = [0]*(math.pow(2,P))
            for i in range(len(rec_lbp)):
                for j in range (len(rec_lbp[0])):
                    lst[int(rec_lbp[i][j])] = lst[int(rec_lbp[i][j])]+1
            normal_lst = map(lambda x:round((float(x)/float(sum(lst))),4),lst)
            out_lst= out_lst+normal_lst
    return np.array(out_lst)

if __name__ == '__main__':
    img= cv2.imread('./1074.jpg',0)
    print img.shape
    LBP_TYPE=('default','ror','uniform','var')
    P = 8
    R = 4
    lbp_type =LBP_TYPE[2]
    out_lst = lbp_like(img,1)
