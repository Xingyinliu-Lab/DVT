import copy
import pandas as pd
import numpy as np
import cv2
import os
import sys
from random import sample
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

fileplace=sys.argv[1]
videoname=sys.argv[2]
background_generate_frames=int(sys.argv[3])
startframe=300
# endframe=int(sys.argv[5])# 2000

codec = 'mp4v'

input_vidpath = fileplace + videoname
modified_vidpath =fileplace+videoname.replace('.','_')+'_cleaned.mp4'
cap = cv2.VideoCapture(input_vidpath)
total_frame=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
endframe=total_frame-100
sample_interval=int(min(total_frame,endframe-startframe)/background_generate_frames)

fps=cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*codec)
x_dim=int(cap.read()[1].shape[0])
y_dim=int(cap.read()[1].shape[1])
output_framesize = (y_dim,x_dim)
font = cv2.FONT_HERSHEY_SIMPLEX

# print(input_vidpath,x_dim,y_dim)

def Contrast_and_Brightness(alpha, beta, img):
    blank = np.zeros(img.shape, img.dtype)
    # dst = alpha * img + beta * blank
    dst = cv2.addWeighted(img, alpha, blank, 1 - alpha, beta)
    return dst


if not os.path.exists(fileplace+videoname.replace('.','_')+'_background.jpg'):
    img_list = []
    last=0
    count=0
    flist=list(np.array(range(startframe,endframe,sample_interval)))

    for i in range(background_generate_frames):
        sample_list=sample(flist,2)
        f1=sample_list[0]
        f2=sample_list[1]
        cap.set(cv2.CAP_PROP_POS_FRAMES, f1)
        ret, frame = cap.read()
        img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        width = int(img1.shape[1])
        height = int(img1.shape[0])
        img_copy=copy.copy(img1)
        img_copy=np.asarray(img_copy,dtype='float')
        cap.set(cv2.CAP_PROP_POS_FRAMES, f2)
        ret, frame = cap.read()
        img2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_diff = cv2.absdiff(img1,img2)
        bright = np.percentile(img_diff, 99.5)
        alpha = 255 / bright
        beta = 0
        img_diff = Contrast_and_Brightness(alpha, beta, img_diff)
        img_copy[img_diff>100]=np.nan
        img_list.append(img_copy)
    img_list=np.asarray(img_list)
    b_list=np.nanmedian(img_list,axis=0)
    mask = np.isnan(b_list)
    idx = np.where(~mask,np.arange(mask.shape[1]),0)
    np.maximum.accumulate(idx,axis=1, out=idx)
    out = b_list[np.arange(idx.shape[0])[:,None], idx]
    b_list=np.asarray(out,dtype='uint8')
    cv2.imwrite(fileplace+videoname.replace('.','_')+'_background.jpg', b_list)
cap.release()

