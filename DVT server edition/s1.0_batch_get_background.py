# python s1.0_batch_get_background.py /data/warehouse/sd427_video_cliped/ 20 3000 50000
import os
import time
import numpy as np
from multiprocessing import Pool
import sys
import pandas as pd
import configparser
conf = configparser.ConfigParser()
conf.read('config.ini')
frames_to_generate_background = conf.get('Fixed_para', 'frames_to_generate_background')

fileplace = sys.argv[1]
processors = int(sys.argv[2])
videolist = [
]

for root, dirs, files in os.walk(fileplace):
    videolist = files
newvideolist = [
]
for file in videolist:
    if not ('.jpg' in file):
        if not ('.csv' in file):
            if ('.mp4' in file):
                newvideolist.append(file)
videolist = newvideolist

# python remove_background_s2.py ND_ND_Male_3_20210105_141226.mp4 video/ 3000 45
def cal_cl(cl):
    for index in cl:
        cmdstr = 'python getBackground_cmd_s1.py ' + fileplace + ' ' + videolist[index] + ' ' + str(
            int(frames_to_generate_background))
        print(cmdstr)
        os.system(cmdstr)
        print(videolist[index] + ' done')
    return None

if __name__ == '__main__':
    processor = processors
    cl = np.array_split(np.asarray(range(len(videolist))), processor, axis=0)
    reslist = []
    p = Pool(processor)
    for i in range(processor):
        reslist.append(p.apply_async(cal_cl, args=(cl[i],)))
    p.close()
    p.join()
    meta = pd.DataFrame(
        columns={'csv', 'genotype', 'background', 'videoname', 'replicate', 'sex', 'num', 'vindex', 'drop', 'x1', 'y1',
                 'x2', 'y2', 'x3', 'y3', 'x', 'y', 'r', 'condition'})
    vcount = 0
    for v in videolist:
        meta.loc[vcount, 'csv'] = v.replace('.', '_') + '_cleaned-position.csv'
        meta.loc[vcount, 'background'] = v.replace('.', '_') + '_background.jpg'
        meta.loc[vcount, 'videoname'] = v
        vcount = vcount + 1
    meta.sort_values(by='videoname', inplace=True)
    meta.reset_index(drop=True, inplace=True)
    meta = meta[
        ['csv', 'genotype', 'condition', 'background', 'videoname', 'replicate', 'sex', 'num', 'vindex', 'drop', 'x1',
         'y1', 'x2', 'y2', 'x3', 'y3', 'x', 'y', 'r']]
    meta.to_csv(fileplace + 'metadata.csv', index=False)
    # rea
