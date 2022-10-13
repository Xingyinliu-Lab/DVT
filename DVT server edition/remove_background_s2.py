import pandas as pd
import numpy as np
import cv2
import os
import sys

# videoname='wt_MF_1_WIN_20201208_20_00_35_Pro.mp4'
fileplace=sys.argv[1]
videoname=sys.argv[2]
x=int(sys.argv[3])
y=int(sys.argv[4])
r=int(sys.argv[5])
pre=sys.argv[6]
new_fileplace=sys.argv[7]

codec = 'mp4v'
input_vidpath = fileplace + videoname
modified_vidpath =new_fileplace+videoname.replace('.','_')+'_cleaned.mp4'
if not os.path.exists(modified_vidpath):

    cap = cv2.VideoCapture(input_vidpath)
    total_frame=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fps=cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    x_dim=int(cap.read()[1].shape[0])
    y_dim=int(cap.read()[1].shape[1])
    output_framesize = (y_dim,x_dim)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # print(input_vidpath,x_dim,y_dim)
    cap.release()

    cap = cv2.VideoCapture(input_vidpath)

    out = cv2.VideoWriter(filename=modified_vidpath, fourcc=fourcc, fps=fps, frameSize=output_framesize, isColor=False)

    b_list=cv2.imread(fileplace+videoname.replace('.','_')+'_background.jpg')
    b_list=cv2.cvtColor(b_list, cv2.COLOR_BGR2GRAY)
    bright=np.mean(b_list)
    # print('brightness: ',bright)
    frames=[]
    last=0
    mask = np.zeros(shape=[int(x_dim),int(y_dim)])
    cv2.circle(mask, (int(x),int(y)), int(r), 255, -1)

    while (True):
        ret, frame = cap.read()
        this = cap.get(1)
        if pre=='1':
            if this%fps!=0:
                continue
        if ret == True:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img_bright=np.mean(img)
            img = np.asarray(img)
            img=np.clip((img+bright-img_bright),0,255)
            img=img.astype('uint8')
            img_diff = cv2.absdiff(b_list, img)
            img_diff[mask < 250] = 0
            out.write(img_diff)
        if last >= this:
            break
        last = this
        if this%(fps*30)==0:
            print(videoname,'remove background @ '+str(this/fps)+'s')
    cap.release()
    out.release()



