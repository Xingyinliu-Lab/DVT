import sys
import cv2
import pandas as pd
import numpy as np

fileplace=sys.argv[1]
metadata_name='metadata.csv'

font = cv2.FONT_HERSHEY_SIMPLEX
metadata=pd.read_csv(fileplace+metadata_name,header=0)
image=None

count=0
for img_index in metadata.index:
    background_name=metadata.loc[img_index,'background']
    img = cv2.imread(fileplace+background_name)
    cv2.putText(img, background_name, (5, 5+30), font, 0.5, (255, 255, 255), 1)
    if not (image is None):
        image = np.vstack((image, img))
    if image is None:
        image=img
    if img_index>0 and img_index%50==0:
        cv2.imwrite(fileplace+'all_background_'+str(count)+'.jpg',image)
        count=count+1
        image=None
if img_index%50!=0:
    cv2.imwrite(fileplace+'all_background_'+str(count)+'.jpg',image)