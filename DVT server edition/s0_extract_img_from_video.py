import pandas as pd
import numpy as np
import cv2
import sys
import os

fileplace=sys.argv[1]
video_place=fileplace
for root, dirs, files in os.walk(fileplace):
    videolist=files
newvideolist=[
]
for file in videolist:
    if not ('.jpg' in file):
        if not ('.csv' in file):
            if ('.mp4' in file):
                newvideolist.append(file)
videolist=newvideolist

N=1
N_file=fileplace+'N_img/'
if not os.path.exists(N_file):
    os.makedirs(N_file)
for v in videolist:
    videoname=v
    cap = cv2.VideoCapture(video_place+videoname)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(100))
    rval, frame = cap.read()
    # img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(N_file+videoname.replace('.','_')+'_'+str(100).zfill(7)+'.jpg',frame)
    cap.release()
