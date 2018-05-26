import os
import random
pos_path = '../sample_generator/classifiy_data/postive'
neg_path = '../sample_generator/negative_sample_folder'
pos_folder = os.listdir(pos_path)
pos_file = './postive_anno.txt'
neg_file = './negtive_anno.txt'
train_file = './train_anno.txt'
test_file = './test_anno.txt'
'''
 this script  follow 2 steps  by flag
 the first stpe is by the pos_path& neg_path get postive&negtive  anno.txt
 the second step is by the two anno txt  get train_anno and test_anno
 '''
pos_flag =False
neg_flag =False
train_flag =True
if pos_flag :
    pos_ctx = open(pos_file,'w')
    for ele in pos_folder:
        #get postive folder
        folder_path = pos_path+os.sep+ele
        img_lst = os.listdir(folder_path)
        for img in img_lst:
            img_url = folder_path+os.sep+img
            ctx = img_url+' '+str(1)+'\n'
            pos_ctx.write(ctx)
    pos_ctx.close()

if neg_flag :
    neg_ctx = open(neg_file,'w')
    for ele in pos_folder:
        print ele,' doing....'
        #get postive folder
        folder_path = neg_path+os.sep+ele
        img_lst = os.listdir(folder_path)
        for img in img_lst:
            img_url = folder_path+os.sep+img
            ctx = img_url+' '+str(0)+'\n'
            neg_ctx.write(ctx)
    neg_ctx.close()

if train_flag :
    train_ctx = open(train_file,'w')
    test_ctx = open(test_file,'w')
    pos_ctx = open(pos_file,'r').readlines()
    neg_ctx = open(neg_file,'r').readlines()
    count_num = len(neg_ctx)

    write_lst = []
    write_lst = write_lst+pos_ctx
    print len(write_lst)
    count = 0
    while(count<count_num):
        if count%10 ==0:
            write_lst.append(neg_ctx[count])
        count = count+1
    part_count = 0
    test_lst = []
    train_lst = []
    for ele in write_lst:
        part_count =part_count+1
        if part_count >10:
            part_count =1
        if part_count<=2:
            test_lst.append(ele)
        else:
            train_lst.append(ele)
    random.shuffle(train_lst)
    random.shuffle(test_lst)
    for train_ele in train_lst:
        train_ctx.write(train_ele)
    train_ctx.close()

    for test_ele in test_lst:
        test_ctx.write(test_ele)
    test_ctx.close()




























