#python s2_batch_remove_background.py /data/warehouse/sd055_clip/ 20 metadata.csv 1
import os
import time
import numpy as np
from multiprocessing import Pool
import sys
import pandas as pd
fileplace=sys.argv[1]
processors=int(sys.argv[2])
metadata_name=sys.argv[3]
pre=sys.argv[4]# 1 for preview 0 for all
new_fileplace=sys.argv[5]

if not os.path.exists(new_fileplace):
    os.mkdir(new_fileplace)

metadata=pd.read_csv(fileplace+metadata_name,header=0)


def cal_cl(cl):
    for index in cl:
        x,y,r=metadata.loc[index,'x'],metadata.loc[index,'y'],metadata.loc[index,'r']
        v_name=metadata.loc[index,'videoname']
        cmdstr='python remove_background_s2.py '+fileplace+' '+v_name+' '+str(int(x))+' '+str(int(y))+' '+str(int(r))+' '+pre+' '+new_fileplace
        print(cmdstr)
        os.system(cmdstr)

    return None


if __name__ == '__main__':
    processor = processors
    cl = np.array_split(np.asarray(metadata.index), processor, axis=0)
    reslist = []
    p = Pool(processor)
    for i in range(processor):
        reslist.append(p.apply_async(cal_cl, args=(cl[i], )))
    p.close()
    p.join()
    # rea


