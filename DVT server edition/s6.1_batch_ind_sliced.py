import os
import time
import numpy as np
from multiprocessing import Pool
import sys
import pandas as pd
# python s1.2_batch_ind.py /data/warehouse/HFD/ metadata.csv 30 1
fileplace=sys.argv[1]
metadata_name=sys.argv[2]
processors=int(sys.argv[3])
predix=sys.argv[4]
scaling=sys.argv[5]

fileplace_analysis=fileplace+'analysis'+predix+'/'
if not os.path.exists(fileplace_analysis):  # 如果文件目录不存在则创建目录
    os.makedirs(fileplace_analysis)

metadata=pd.read_csv(fileplace+metadata_name,header=0)

# python remove_background_s2.py ND_ND_Male_3_20210105_141226.mp4 video/ 3000 45
def cal_cl(cl):
    for idindex in cl:
        cmdstr='python s1.1_ind_motion_avg.py '+fileplace+' '+metadata_name+' '+str(idindex)+' '+predix+' '+scaling
        print(cmdstr)
        os.system(cmdstr)
    return None
#
# fileplace_res = sys.argv[1]
# id_name = sys.argv[2]
# type_name = sys.argv[3]




if __name__ == '__main__':
    processor = processors
    cl = np.array_split(np.asarray(metadata.index), processor, axis=0)
    reslist = []
    p = Pool(processor)
    for i in range(processor):
        reslist.append(p.apply_async(cal_cl, args=(cl[i],)))
    p.close()
    p.join()

    # rea


