import sys
import os
import configparser
# s99.4 motion and interaction analysis
#python s99.4_batchscript.py 48 /data/warehouse/VT0730_480p/ 0 0
processors = sys.argv[1]
fileplace=sys.argv[2]
prefix=sys.argv[3]
annotated=sys.argv[4]

if annotated=='1':
    annotated_video_place=sys.argv[5]
else:
    annotated_video_place =''



conf=configparser.ConfigParser()
conf.read('config.ini')
scaling=conf.get('Fixed_para','scaling')



# python s1.2_batch_ind.py /data/warehouse/HFD/ metadata.csv 30 1

cmdstr='python s6_batch_ind.py '+fileplace+' metadata.csv '+processors+' '+prefix+' '+scaling
os.system(cmdstr)
# python s2.2_batch_group.py /data/warehouse/HFD/ metadata.csv 30 1
cmdstr='python s7_batch_group.py '+fileplace+' metadata.csv '+processors+' '+prefix+' '+scaling
os.system(cmdstr)

cmdstr='python s6.1_batch_ind_sliced.py '+fileplace+' metadata.csv '+processors+' '+prefix+' '+scaling
os.system(cmdstr)
# python s2.2_batch_group.py /data/warehouse/HFD/ metadata.csv 30 1
cmdstr='python s7.1_batch_group_sliced.py '+fileplace+' metadata.csv '+processors+' '+prefix+' '+scaling
os.system(cmdstr)


cmdstr='python s8_stat.py '+fileplace+' '+prefix
os.system(cmdstr)

if annotated=='1':
    cmdstr='python s9_batch_annotated.py '+annotated_video_place+' metadata.csv '+processors+' '+fileplace
    os.system(cmdstr)

cmdstr='python s10_plot.py '+fileplace+' '+prefix
os.system(cmdstr)

cmdstr='python s10.1_plot_sliced.py '+fileplace+' '+prefix
os.system(cmdstr)






# according  to che error check csv fixed the errors