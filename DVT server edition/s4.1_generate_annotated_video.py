import cv2
import numpy as np
import pandas as pd
import sys
import configparser
conf=configparser.ConfigParser()
conf.read('config.ini')
scaling =float(conf.get('Fixed_para','scaling'))

fileplace=sys.argv[1]
metadata_name=sys.argv[2]
meta_index=int(sys.argv[3])
csv_fileplace=sys.argv[4]

metadata=pd.read_csv(fileplace+metadata_name,header=0)
csv_name=metadata.loc[meta_index,'csv']
v_name=metadata.loc[meta_index,'videoname']
saveplace=fileplace
tracefile=csv_fileplace+csv_name
videoname=fileplace+v_name
n_inds = metadata.loc[meta_index,'num']


colorlist = [(60, 0, 255),
             (140, 0, 255),
             (220, 0, 255),
             (255, 0, 200),
             (255, 0, 120),
             (255, 0, 40),
             (255, 20, 0),
             (255, 100, 0),
             (255, 180, 0),
             (240, 255, 0),
             (160, 255, 0),
             (80, 255, 0),
             (0, 255, 0),
             (0, 255, 60),
             (0, 255, 140),
             (0, 255, 220)]


def annoateVideo(tracefile,videoname,n_inds):
    video=videoname
    outvideo=videoname.replace('.','_')+'_annoated_video.mp4'
    trace=tracefile
    trace=pd.read_csv(trace,header=0,index_col=False)
    cap = cv2.VideoCapture(video)
    n_inds = int(n_inds)
    subject_list= ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H','I','J','K','L','M','N']
    xlabel_list=[]
    ylabel_list=[]
    for i in range(n_inds):
        xlabel_list.append('x'+str(int(i)))
        ylabel_list.append('y'+str(int(i)))
    codec = 'mp4v'
    fps=cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    x_dim=int(cap.read()[1].shape[0])
    y_dim=int(cap.read()[1].shape[1])
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale=scaling
    output_framesize = (y_dim,x_dim)
    red = (48, 48, 255)
    green = (34, 139, 34)
    yellow = (0, 255, 255)
    out = cv2.VideoWriter(filename=outvideo, fourcc=fourcc, fps=fps, frameSize=output_framesize, isColor=True)
    last=0
    while True:
        res, image = cap.read()
        this = int(cap.get(1))
        if res == True:
            this_list=np.array(range(max(0,this-15),this+1))
            tmp_trace = trace[trace['position'].isin(this_list)]
            tmp_trace = tmp_trace.reset_index(drop=True)
            if len(tmp_trace)==0:
                continue
            for i in range(n_inds):
                x_label=xlabel_list[i]
                y_label = ylabel_list[i]
                for j in tmp_trace.index:
                    x = int(tmp_trace.loc[j,x_label]*scale)
                    y = int(tmp_trace.loc[j, y_label]*scale)
                    cv2.circle(image, (x,y), 2, colorlist[j], -1, cv2.LINE_AA)
                cv2.putText(image, subject_list[i], (x+5,y+15), font, 0.5, green, 1)
            cv2.putText(image, str(int(this)), (5, 30), font, 1, (255, 255, 255), 2)
            out.write(image)
        if last >= this:
            break
        last = this
        # if this%180==0:
        #     print(this/30)
    print(videoname+' traced video generation finished')
    cap.release()
    out.release()
    return None

annoateVideo(tracefile,videoname,n_inds)