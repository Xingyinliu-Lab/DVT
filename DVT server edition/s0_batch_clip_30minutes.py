


#python s0_batch_clip_30minutes.py /data/warehouse/v422/ 12 /data/warehouse/v422_clip/ 55 1855
import os
import time
import numpy as np
from multiprocessing import Pool
import sys
fileplace=sys.argv[1]
processors=int(sys.argv[2])
cliped_fileplace=sys.argv[3]
starttime=sys.argv[4]
endtime=sys.argv[5]
# fileplace='G:\\fly_movie\\210105_HFD_7day\\480p\\video/'
videolist=[
]

if not os.path.exists(cliped_fileplace):
    os.mkdir(cliped_fileplace)
for root, dirs, files in os.walk(fileplace):
    videolist=files

newvideolist=[
]
for file in videolist:
    if not ('.jpg' in file):
        newvideolist.append(file)
videolist=newvideolist

# python remove_background_s2.py ND_ND_Male_3_20210105_141226.mp4 video/ 3000 45
def cal_cl(cl):
    for index in cl:
        cmdstr='python clip_video_for_30minutes_s0.py '+fileplace+' '+videolist[index]+' '+cliped_fileplace+' '+starttime+' '+endtime
        print(cmdstr)
        os.system(cmdstr)
        print(videolist[index]+' done')
    return None

# videoname=sys.argv[1]
# fileplace=sys.argv[2]
# newfileplace=sys.argv[3]

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


