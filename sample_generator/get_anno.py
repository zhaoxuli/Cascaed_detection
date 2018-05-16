import os
import json as js
#get txt Anno from Json file
img_path = '/mnt/hgfs/linux_share/phone_data/Image'
anno_path = '/mnt/hgfs/linux_share/phone_data/Anno'
anno_txt = '/mnt/hgfs/linux_share/phone_data/Anno_txt'

img_lst= os.listdir(img_path)


for ele in img_lst:
    folder_path = img_path+os.sep+ele
    anno_file = anno_path+os.sep+ele+'.json'
    if  os.path.exists(anno_file)==False:
        print 'the '+ele +' have not anno file'
    else:
        lines = open(anno_file,'r').readlines()
        txt_path = anno_txt+os.sep+ele+'.txt'
        txt_ctx = open(txt_path,'w')
        for line in lines:
            ctx = js.loads(line)
            img_key = ctx['image_key']
            try:
                rect_gt= ctx['common_box'][0]['data']
                left_point = [rect_gt[0],rect_gt[1]]
                right_point = [rect_gt[2],rect_gt[3]]
                write_ctx = str(img_key) +' '+str(left_point)+' '+str(right_point)+'\n'
                txt_ctx.write(write_ctx)
            except:
                print 'the '+img_key+' have not gt info'
        txt_ctx.close


