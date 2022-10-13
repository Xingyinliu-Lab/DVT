from moviepy.editor import *
import sys
import configparser
conf=configparser.ConfigParser()
conf.read('config.ini')
fps =int(conf.get('Fixed_para','fps'))

fileplace=sys.argv[1]
videoname=sys.argv[2]

newfileplace=sys.argv[3]
start_time=int(sys.argv[4])
end_time=int(sys.argv[5])

clip=VideoFileClip(fileplace+videoname)
video_duration=int(clip.duration)

end_time=min(end_time,video_duration-10)

clip=clip.subclip(start_time,end_time)
videoname=videoname.replace('.avi','.mp4')
videoname=videoname.replace('.mkv','.mp4')
if not ('.mp4' in videoname):
    videoname=videoname+'.mp4'

clip.to_videofile(newfileplace+videoname, fps=fps)
