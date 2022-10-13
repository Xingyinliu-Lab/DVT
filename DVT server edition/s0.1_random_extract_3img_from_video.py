import cv2
import sys
import os
import configparser
import pandas as pd
conf=configparser.ConfigParser()
conf.read('config.ini')
imgs_for_body_size_measure =int(conf.get('Fixed_para','imgs_for_body_size_measure'))

fileplace=sys.argv[1]
meta='metadata.csv'
metadata=pd.read_csv(fileplace+meta,header=0)

videolist=list(metadata['videoname'])

N_file=fileplace+'img_for_body_size_measure/'

if not os.path.exists(N_file):
    os.makedirs(N_file)
import random
for v in videolist:
    videoname=v
    cap = cv2.VideoCapture(fileplace+videoname)
    frames_num=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(imgs_for_body_size_measure):
        f=random.randint(1,int(frames_num))
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(f))
        rval, frame = cap.read()
        cv2.imwrite(N_file+videoname.replace('.','_')+'_'+str(f).zfill(7)+'.jpg',frame)
    cap.release()
