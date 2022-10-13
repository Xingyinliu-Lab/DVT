import sys
import os
#python s99.2_batchscript.py 24 /data/warehouse/VT0730_clip/ /data/warehouse/VT0730_clean/ /data/warehouse/VT0730_480p/
# s99.2 remove background and scale video to 480p
processors = sys.argv[1]
clipedvideo_fileplace=sys.argv[2]
cleaneddvideo_fileplace=sys.argv[3]
video480p_fileplace=sys.argv[4]


cmdstr='python s1.2_find_cycle.py '+clipedvideo_fileplace
os.system(cmdstr)

#python s2_batch_remove_background.py /data/warehouse/sd055_clip/ 20 metadata.csv 1
cmdstr='python s2_batch_remove_background.py '+clipedvideo_fileplace+' '+processors+' metadata.csv 0 '+cleaneddvideo_fileplace
os.system(cmdstr)

cmdstr='python s4_batch_to_480.py '+cleaneddvideo_fileplace+' '+processors+' '+video480p_fileplace+' '+clipedvideo_fileplace
os.system(cmdstr)
# mannualy use UMA to track drosophila