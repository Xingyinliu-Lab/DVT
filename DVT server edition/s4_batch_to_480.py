import os
import time
import numpy as np
import pandas as pd
from multiprocessing import Pool
import sys
fileplace=sys.argv[1]
processors=int(sys.argv[2])
newfileplace=sys.argv[3]
clipedvideo_fileplace=sys.argv[4]

if not os.path.exists(newfileplace):
    os.mkdir(newfileplace)

videolist=[
]

for root, dirs, files in os.walk(fileplace):
    videolist=files
videolist2=[
]
for vi in videolist:
    if '.mp4' in vi:
        videolist2.append(vi)
videolist=videolist2

metadata=pd.read_csv(clipedvideo_fileplace+'metadata.csv',header=0)
metadata.to_csv(newfileplace+'metadata.csv',index=False)

def cal_cl(cl):
    for index in cl:
        if '.mp4' in videolist[index]:
            cmdstr='python 768_to_480_s4.py '+videolist[index]+' '+fileplace+' '+newfileplace
            print(cmdstr)
            os.system(cmdstr)
            print(videolist[index]+' done')
    return None

if __name__ == '__main__':
    processor = processors
    cl = np.array_split(np.asarray(range(len(videolist))), processor, axis=0)
    reslist = []
    p = Pool(processor)
    for i in range(processor):
        reslist.append(p.apply_async(cal_cl, args=(cl[i], )))
    p.close()
    p.join()
    # rea


