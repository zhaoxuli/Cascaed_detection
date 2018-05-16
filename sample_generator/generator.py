# -*- coding: utf-8 -*-

import cv2
import os
global out_img
global img_folder
def do_resize(img_path,gt_lst,size):
    img =cv2.imread(img_path,0)
    #y,x=img.shape  #x is row(lie) y is col(hang)
    img_size = img.shape
    y,x=[img_size[0],img_size[1]]
    dst = cv2.resize(img,(size,size))
    scale_x = float(size)/float(x)  #height
    scale_y = float(size)/float(y)   #width
    lx = int(scale_x*gt_lst[0])
    ly = int(scale_y*gt_lst[1])
    rx = int(scale_x*gt_lst[2])
    ry = int(scale_y*gt_lst[3])

    re_gt_lst = [lx,ly,rx,ry]
    return dst,re_gt_lst

def  generat_pyramid_dst(img,gt_lst,t_size):
    y,x =[img.shape[0],img.shape[1]]
    dst = cv2.resize(img,(t_size,t_size))
    scale_x = float(t_size)/float(x)  #height
    scale_y = float(t_size)/float(y)   #width
    lx = int(scale_x*gt_lst[0])
    ly = int(scale_y*gt_lst[1])
    rx = int(scale_x*gt_lst[2])
    ry = int(scale_y*gt_lst[3])

    re_gt_lst = [lx,ly,rx,ry]
    return dst,re_gt_lst


def sample_generator(dst,gt_lst,ref_type,filter_size,norm_len,
                     min_norm_len,positive_boundary,regression_boundary,
                     positive_pick_ratio,regression_pick_ratio):
    global img_folder
    global out_img
    count = 0
    boundary = regression_boundary
    #juge
    if ref_type =='height':
        value = gt_lst[3]-gt_lst[1]
    if ref_type =='width':
        value = gt_lst[2]-gt_lst[0]
    if value <min_norm_len :
        return
    #value is the thruth object lenth
    ref_w,ref_h =filter_size
    size = dst.shape[0]
    #get pyramid_size
    pyramid_size_factor = (size*norm_len)/value
    pyramid_size = [boundary[0]*pyramid_size_factor,boundary[1]*pyramid_size_factor]
    print 'pyramid_size:',pyramid_size
    print 'new_size:',
    for t_size in range(int(pyramid_size[0]),int(pyramid_size[1]),5):
        scale = float(t_size)/float(size)
        print 'p_scale:',scale
        boundary_ratio= value*scale/norm_len
        #if (boundary_ratio in range(float(boundary[0]),float(boundary[1]))
        print 'boundary_ratio,boundary,norm_len*P-scale:',boundary_ratio,boundary,value*scale
        if (float(boundary[0])<boundary_ratio<float(boundary[1]) and value*scale >min_norm_len):
             prmd_dst,prmd_gt_lst=generat_pyramid_dst(dst,gt_lst,t_size)
             #generator
             lx,ly,rx,ry =prmd_gt_lst
             print prmd_gt_lst
             mid_x,mid_y = [lx+(rx-lx)/2,ly+(ry-ly)/2]
             pick_ratio = [regression_pick_ratio,positive_pick_ratio]
             sample_folder = ['./generator_data/regression_sample_folder/','./generator_data/positive_sample_folder/']
             #Attention the first element is regression ratio
             i = 0
             while(i<2):
                 save_folder = sample_folder[i]+img_folder
                 if  os.path.exists(save_folder) == False:
                       os.makedirs(save_folder)
                 if i==0:
                     Anno_file = save_folder+'.txt'
                     f=open(Anno_file,'a')
                 lenth = int(pick_ratio[i]*(min(rx-lx,ry-ly))/2)
                 range_x = [int(mid_x-lenth),int(mid_x+lenth)]
                 range_y = [int(mid_y-lenth),int(mid_y+lenth)]
                 step_dic={'mild':1,'stander':2,'wider':3}
                 step = step_dic['stander']
                 #show(prmd_dst,str(t_size)+' '+str(scale),prmd_gt_lst)
                 for x_center in range(range_x[0],range_x[1],step):
                     for y_center in range(range_y[0],range_y[1],step):
                        pick_x =[int(x_center-ref_w/2),int(x_center+ref_w/2)]
                        pick_y =[int(y_center-ref_h/2),int(y_center+ref_h/2)]
                        ref_dst = prmd_dst[pick_y[0]:pick_y[1],pick_x[0]:pick_x[1]]
                        lst = [pick_x[0],pick_y[0],pick_x[1],pick_y[1]]
                        out_img_path = save_folder +os.sep+out_img+str(count)+'.jpg'
                        cv2.imwrite(out_img_path,ref_dst)
                        if i==0:
                            #print out_img_path
                            res_lst = [lx-lst[0],ly-lst[1],rx-lst[2],ry-lst[3]]
                            ctx = out_img_path +' '+str(res_lst) +'\n'
                            f.write(ctx)
                            f.close
                        count = count+1
                 i=i+1

def generate_negative_sample(dst,gt_lst,ref_type,filter_size,norm_len,min_norm_len,negative_boundary,negative_pick_ratio):
    global img_folder
    global out_img
    count = 0
    save_folder = './generator_data/negative_sample_folder/'+img_folder
    if  os.path.exists(save_folder) == False:
          os.makedirs(save_folder)

    ref_w,ref_h = filter_size
    prymd_factor=[0.25,0.5,1,2,4]
    img_size = dst.shape[0]
    if ref_type =='height':
        value = gt_lst[3]-gt_lst[1]
    if ref_type =='width':
        value = gt_lst[2]-gt_lst[0]
    if value <min_norm_len :
        return
    for factor in prymd_factor:
        new_size = int(img_size*factor)
        prmd_dst = cv2.resize(dst,(new_size,new_size))
        boundary_ratio =float(factor*value/norm_len)
        center_x_range = [int(ref_w/2),int(new_size-ref_w/2)]
        center_y_range = [int(ref_h/2),int(new_size-ref_h/2)]
        #step_dic={'mild':2,'stander':3,'wider':5}
        step = int(new_size/10)

        prmd_lx = int(gt_lst[0]*factor)
        prmd_ly = int(gt_lst[1]*factor)
        prmd_rx = int(gt_lst[2]*factor)
        prmd_ry = int(gt_lst[3]*factor)

        lenth = int(negative_pick_ratio*(min(prmd_rx-prmd_lx,prmd_ry-prmd_ly)/2))
        mid_x,mid_y = [new_size/2,new_size/2]
        ignore_x_range = [mid_x-lenth/2,mid_x+lenth/2]
        ignore_y_range = [mid_y-lenth/2,mid_y+lenth/2]

        for x_center  in range(center_x_range[0],center_x_range[1],step):
            for y_center  in range(center_y_range[0],center_y_range[1],step):
                pick_x =[int(x_center-ref_w/2),int(x_center+ref_w/2)]
                pick_y =[int(y_center-ref_h/2),int(y_center+ref_h/2)]
                if (negative_boundary[0]<boundary_ratio<negative_boundary[1]):
                    if (ignore_x_range[0]<=x_center<=ignore_x_range[1] and ignore_y_range[0]<=y_center<=ignore_y_range[1]):
                        print 'ignore_samples'
                    else:
                        ref_dst = prmd_dst[pick_y[0]:pick_y[1],pick_x[0]:pick_x[1]]
                else:
                     ref_dst = prmd_dst[pick_y[0]:pick_y[1],pick_x[0]:pick_x[1]]
#                cv2.line(prmd_dst,(0,y_center),(new_size,y_center),255,1)
#                cv2.line(prmd_dst,(x_center,0),(x_center,new_size),255,1)
#                print 'center_x_range:',center_x_range,'   ',
#                print 'center_y_range:',center_y_range
#                print 'x_center,y_center:',x_center,y_center
                #show(prmd_dst,title=str(new_size))
                if ref_dst.shape != None:
                    out_img_path = save_folder+os.sep+out_img+str(count)+'.jpg'
                    cv2.imwrite(out_img_path,ref_dst)
                count =count+1



def show(dst,title='default',gt_lst=None):
    if gt_lst is not None:
        lx,ly,rx,ry =gt_lst
        cv2.rectangle(dst,(lx,ly),(rx,ry),(0,255,0),1)
    cv2.imshow(title,dst)
    k = chr(cv2.waitKey() & 255)
    if k == 'q' :
        quit()
    elif k=='n':
        return


def run(Paramter):
    #load Paramter
    Image_path  =Paramter['Image_path']
    Anno_path   =Paramter['Anno_path']
    size        =Paramter['resize']
    ref_type    =Paramter['ref_type']
    filter_size = [Paramter['ref_w'],Paramter['ref_h']]
    norm_len    =Paramter['norm_len']
    min_norm_len=Paramter['min_vail_len']
    positive_boundary    =[Paramter['positive_low_boundary'],Paramter['positive_up_boundary']]
    regression_boundary    =[Paramter['regression_low_boundary'],Paramter['regression_up_boundary']]
    negative_boundary    =[Paramter['ignore_low_boundary'],Paramter['ignore_up_boundary']]

    positive_pick_ratio  =Paramter['postive_gener_ratio']
    regression_pick_ratio  =Paramter['regression_gener_ratio']
    negative_pick_ratio  =Paramter['ignore_gener_ratio']

    #load image
    Anno_lst = os.listdir(Anno_path)
    for  anno in Anno_lst:
        Anno_file=Anno_path+os.sep+anno
        anno_ctx = open(Anno_file).readlines()
        global img_folder
        img_folder = Anno_file.split(os.sep)[-1][:-4]
        for ele in anno_ctx:
            info_lst = ele.split('[')
            img_key = info_lst[0][:-1]
            img_path = Image_path+os.sep+img_folder+os.sep+img_key
            global out_img
            out_img = img_key.split('.')[0]
            if os.path.exists(img_path):
                gt_lst = []
                left_x =int(info_lst[1].split(' ')[0][:-1])
                left_y =int(info_lst[1].split(' ')[1][:-1])
                right_x=int(info_lst[2].split(' ')[0][:-1])
                right_y=int(info_lst[2].split(' ')[1][:-2])# '\n' in the last
                gt_lst = [left_x,left_y,right_x,right_y]
                dst,re_gt_lst =do_resize(img_path,gt_lst,size)
                #print dst.shape,gt_lst
                sample_generator(dst,re_gt_lst,ref_type,filter_size,norm_len,min_norm_len,
                                 positive_boundary,regression_boundary,positive_pick_ratio,regression_pick_ratio)
                generate_negative_sample(dst,gt_lst,ref_type,filter_size,norm_len,min_norm_len,negative_boundary,negative_pick_ratio)


