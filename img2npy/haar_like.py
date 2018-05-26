import numpy as np
import cv2

SHAPE_TYPE = ('edge_x','edge_y','line_x','line_y','center','diagonal')

def Integral_feature(img):
    H,W = img.shape
    Iteral = np.zeros([H,W])
    Iteral[0,0]=img[0,0]
    for w in range(1,W):
        Iteral[0,w] = Iteral[0,w-1] +img[0,w]
    for h in range(1,H):
        Iteral[h,0] = Iteral[h-1,0] +img[h,0]
    for w in range(1,W):
        for h in range(1,H):
            Iteral[h,w] = Iteral[h-1,w] + Iteral[h,w-1] - Iteral[h-1,w-1] + img[h,w]
    return Iteral

def  haar_filter(Iteral,filter_h,filter_w,shape_type,step=1,start_x=0,start_y=0):
    if shape_type not in SHAPE_TYPE:
        return  None
    else:
        out = np.zeros(1)
        H,W = Iteral.shape
        for h in range(start_y,H,step):
            for w in range(start_x,W,step):
                if shape_type == 'edge_x':
                    '''Horizion filter  like this:
                        1,1     1,2
                        2,1     2,2
                        3,1     3,2
                    '''
                    x1,y1 = w,h
                    x2 = min(x1+ filter_w,W-1)
                    y2 = min(y1+ filter_h/2,H-1)
                    y3 = min(y1+y1+ filter_h,H-1)
                    white = Iteral[y2,x2] - Iteral[y1,x2] - Iteral[y2,x1] + Iteral[y1,x1]
                    black = Iteral[y3,x2] - Iteral[y2,x2] - Iteral[y3,x1] + Iteral[y2,x1]
                    out = np.append(out,white-black)
                if shape_type == 'edge_y':
                    ''' Vertical  filter like this:
                        1,1     1,2     1,3
                        2,1     2,2     2,3
                    '''
                    x1,y1 = w,h
                    x2 = min(x1+ filter_w/2,W-1)
                    x3 = min(x1+ filter_w  ,W-1)
                    y2 = min(y1+ filter_h  ,H-1)
                    white = Iteral[y2,x2] - Iteral[y1,x2] - Iteral[y2,x1] +Iteral[y1,x1]
                    black = Iteral[y2,x3] - Iteral[y2,x2] - Iteral[y1,x3] + Iteral[y1,x2]
                    out = np.append(out,white-black)

                if shape_type == 'line_x':
                    ''' Horizion_line filter like this:
                        1,1    1,2
                        2,1    2,2
                        3,1    3,2
                        4,1    4,2
                    '''
                    x1,y1 = w,h
                    x2 = min(filter_w + x1 , W-1)
                    y2 = min(filter_h/3 +y1, H-1)
                    y3 = min(filter_h/3*2 + y1,H-1)
                    y4 = min(filter_h+ y1 , H-1)
                    big_white = Iteral[y4,x2] - Iteral[y4,x1] - Iteral[y1,x2] + Iteral[y1,x1]
                    double_black = 2*(Iteral[y3,x2] - Iteral[y3,x1] - Iteral[y2,x2]+ Iteral[y2,x1])
                    out = np.append(out,big_white-double_black)

                if shape_type == 'line_y':
                    ''' Vertical_line filter like this:
                        1,1    1,2    1,3    1,4
                        2,1    2.2    2,3    2,4

                    '''
                    x1,y1 = w,h
                    y2 = min(filter_h + y1 , H-1)
                    x2 = min(filter_w/3 +x1, W-1)
                    x3 = min(filter_w/3*2 + x1,W-1)
                    x4 = min(filter_w+ x1 , W-1)
                    big_white = Iteral[y2,x4] - Iteral[y1,x4] - Iteral[y2,x1] + Iteral[y1,x1]
                    double_black = 2*(Iteral[y2,x3] - Iteral[y2,x2] - Iteral[y1,x3] + Iteral[y1,x2])
                    out = np.append(out,big_white-double_black)

                if shape_type == 'center':
                    '''Center filter like this:
                        1,1     1,2     1,3     1,4
                        2,1     2,2     2,3     2,4
                        3,1     3,2     3,3     3,4
                        4,1     4,2     4,3     4,4
                    '''
                    x1,y1 = w,h
                    x2 = min(filter_w/3 +x1, W-1)
                    x3 = min(filter_w/3*2 + x1,W-1)
                    x4 = min(filter_w+ x1 , W-1)
                    y2 = min(filter_h/3 +y1, H-1)
                    y3 = min(filter_h/3*2 + y1,H-1)
                    y4 = min(filter_h+ y1 , H-1)
                    big_white = Iteral[y4,x4] - Iteral[y4,x1] -Iteral[y1,x4] + Iteral[y1,x1]
                    double_black = 2*(Iteral[y3,x3] - Iteral[y3,x2] -Iteral[y2,x3] + Iteral[y2,x2])
                    out = np.append(out,big_white-double_black)

                if shape_type == 'diagonal':
                    '''diagonal filter like this:
                        1,1     1,2     1,3
                        2,1     2,2     2,3
                        3,1     3,2     3,3
                    '''
                    x1,y1 = w,h
                    x2 = min(x1+ filter_w/2,W-1)
                    x3 = min(x1+ filter_w  ,W-1)
                    y2 = min(y1+ filter_h/2,H-1)
                    y3 = min(y1+y1+ filter_h,H-1)
                    big_white = Iteral[y3,x3] - Iteral[y3,x1]-Iteral[y1,x3] +Iteral[y1,x1]
                    double_black =2*(Iteral[y3,x2] - Iteral[y3,x1] - Iteral[y2,x2] + Iteral[y2,x1] +
                                     Iteral[y2,x3] - Iteral[y2,x2] - Iteral[y1,x3] + Iteral[y1,x2])
                    out = np.append(out,big_white-double_black)

        return out[1:]

if __name__ == '__main__':
    img = cv2.imread('./test.jpg',0)
    Iteral = Integral_feature(img)
    out = haar_filter(Iteral,8,12,'edge_y',step =2)
    print out
    print out.shape

