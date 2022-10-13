from moviepy.editor import *
import configparser
conf=configparser.ConfigParser()
conf.read('config.ini')
fps =int(conf.get('Fixed_para','fps'))
resized_x = int(conf.get('Fixed_para','resized_x'))
resized_y = int(conf.get('Fixed_para','resized_y'))

import sys
import os
videoname=sys.argv[1]
fileplace=sys.argv[2]
newfileplace=sys.argv[3]

if not os.path.exists(newfileplace+videoname):
    clip=VideoFileClip(fileplace+videoname).resize((resized_x,resized_y))
    clip.to_videofile(newfileplace+videoname, fps=int(fps))
    clip.close()