import os
import time
import numpy as np
from multiprocessing import Pool
import sys
import pandas as pd
# python s4.2_batch_annotated.py /data/warehouse/768p/ metadata.csv 20
fileplace=sys.argv[1]
metadata_name=sys.argv[2]
processors=int(sys.argv[3])
csv_fileplace=sys.argv[4]

metadata=pd.read_csv(fileplace+metadata_name,header=0)
# metadata=metadata[metadata['num']>1]


def cal_cl(cl):
    for idindex in cl:
        cmdstr='python s4.1_generate_annotated_video.py '+fileplace+' '+metadata_name+' '+str(idindex)+' '+csv_fileplace
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


