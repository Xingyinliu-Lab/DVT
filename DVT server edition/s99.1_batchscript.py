import sys
import os
#python s99.1_batchscript.py 48 /data/warehouse/VT0730/ /data/warehouse/VT0730_clip/ 75 1875

# s99.1 clip video and generate background and combine background
processors = sys.argv[1]
rawvideo_fileplace=sys.argv[2]
clipedvideo_fileplace=sys.argv[3]
clipedvideo_start=sys.argv[4]
clipedvideo_end=sys.argv[5]

cmdstr='python s0_batch_clip_30minutes.py '+rawvideo_fileplace+' '+processors+' '+clipedvideo_fileplace+' '+clipedvideo_start+' '+clipedvideo_end
os.system(cmdstr)
#python s1.0_batch_get_background.py /data/warehouse/sd427_video_cliped/ 20
cmdstr='python s1.0_batch_get_background.py '+clipedvideo_fileplace+' '+processors
os.system(cmdstr)

cmdstr='python s1.1_generate_all_background.py '+clipedvideo_fileplace
os.system(cmdstr)

cmdstr='python s0_extract_img_from_video.py '+clipedvideo_fileplace
os.system(cmdstr)

cmdstr='python s0.1_random_extract_3img_from_video.py '+clipedvideo_fileplace
os.system(cmdstr)

# mannually find cycle in the background and fill out the metadata.csv