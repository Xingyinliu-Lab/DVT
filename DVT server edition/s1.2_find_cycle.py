import sys
import cv2
import pandas as pd
import numpy as np
fileplace=sys.argv[1]
metadata_name='metadata.csv'

def find_center(A,B,C):
    # A = np.array([2.0, 1.5])
    # B = np.array([6.0, 4.5])
    # C = np.array([11.75, 6.25])
    a = np.linalg.norm(C - B)
    b = np.linalg.norm(C - A)
    c = np.linalg.norm(B - A)
    s = (a + b + c) / 2
    R = a*b*c / 4 / np.sqrt(s * (s - a) * (s - b) * (s - c))
    b1 = a*a * (b*b + c*c - a*a)
    b2 = b*b * (a*a + c*c - b*b)
    b3 = c*c * (a*a + b*b - c*c)
    P = np.column_stack((A, B, C)).dot(np.hstack((b1, b2, b3)))
    P /= b1 + b2 + b3
    return P,R


font = cv2.FONT_HERSHEY_SIMPLEX
metadata=pd.read_csv(fileplace+metadata_name,header=0)
metadata['x']=0
metadata['y']=0
metadata['r']=0
image=None

count=0

for img_index in metadata.index:
    A=np.array([metadata.loc[img_index,'x1'],metadata.loc[img_index,'y1']])
    B=np.array([metadata.loc[img_index,'x2'],metadata.loc[img_index,'y2']])
    C=np.array([metadata.loc[img_index,'x3'],metadata.loc[img_index,'y3']])
    P,R=find_center(A,B,C)
    metadata.loc[img_index,'x']=P[0]
    metadata.loc[img_index,'y']=P[1]
    metadata.loc[img_index,'r']=R
    background_name=metadata.loc[img_index,'background']
    img = cv2.imread(fileplace+background_name)

    cv2.putText(img, background_name, (5, 5+30), font, 0.5, (255, 255, 255), 1)
    cv2.circle(img, (int(metadata.loc[img_index,'x']),int(metadata.loc[img_index,'y'])), int(R), (255, 255, 255), 1)
    if not (image is None):
        image = np.vstack((image, img))
    if image is None:
        image=img
    print(img_index)
    if img_index>0 and img_index%50==0:
        cv2.imwrite(fileplace+'cycled_all_background_'+str(count)+'.jpg',image)
        count=count+1
        image=None
if img_index%50!=0:
    cv2.imwrite(fileplace+'cycled_all_background_'+str(count)+'.jpg',image)
metadata = metadata[
    ['csv', 'genotype', 'condition', 'background', 'videoname', 'replicate', 'sex', 'num', 'vindex', 'drop', 'x1',
     'y1', 'x2', 'y2', 'x3', 'y3', 'x', 'y', 'r']]
metadata.to_csv(fileplace+metadata_name)