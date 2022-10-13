import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import cv2
import sys
from sklearn.metrics import r2_score
from scipy import stats
from numpy.lib.stride_tricks import as_strided as stride
# from statsmodels.robust.scale import huber
import warnings
warnings.filterwarnings("ignore")
# from scipy import sparse

fileplace=sys.argv[1]
metadata_name=sys.argv[2]
meta_index=int(sys.argv[3])
predix=sys.argv[4]# 1 for 10 min; 2for 20 min; 3 for 30 min;4 for 40 min;5 for 50 min; 0 for all
scaling=float(sys.argv[5])
#
# fileplace='D:\\t/'
# metadata_name='metadata.csv'
# meta_index=4
# predix='0'# 1 for 10 min; 2for 20 min; 3 for 30 min;4 for 40 min;5 for 50 min; 0 for all
# scaling=1.6



fileplace_analysis=fileplace+'analysis'+predix+'/'
if not os.path.exists(fileplace_analysis):     #如果文件目录不存在则创建目录
    os.makedirs(fileplace_analysis)
import configparser
conf = configparser.ConfigParser()
conf.read('config.ini')
real_r = float(conf.get('Fixed_para', 'diameter'))/2
x_max=int(conf.get('Fixed_para', 'x'))
y_max=int(conf.get('Fixed_para', 'y'))
fps=int(conf.get('Fixed_para', 'fps'))
x_min=0
y_min=0
r_thresh=float(conf.get('Adjustable_para', 'r_thresh'))#mm distance to edge
r_thresh=r_thresh/real_r
max_v_thresh=float(conf.get('Adjustable_para', 'max_v_thresh'))
area_thresh=float(conf.get('Adjustable_para', 'area_thresh'))
sensing_area=float(conf.get('Adjustable_para', 'sensing_area')) # in pixel
move_thresh=float(conf.get('Adjustable_para', 'move_thresh'))
area_search_time_thresh=float(conf.get('Adjustable_para', 'area_search_time_thresh'))
long_stop_thresh=float(conf.get('Adjustable_para', 'long_stop_thresh'))
angular_velocity_window=float(conf.get('Adjustable_para', 'angular_velocity_window'))
track_straightness_window=float(conf.get('Adjustable_para', 'track_straightness_window'))

metadata=pd.read_csv(fileplace+metadata_name,header=0)
csv_name=metadata.loc[meta_index,'csv']
v_name=metadata.loc[meta_index,'videoname']
x,y,r=metadata.loc[meta_index,'x'],metadata.loc[meta_index,'y'],metadata.loc[meta_index,'r']
cycle_area=np.pi*r*r
n_inds = int(float(metadata.loc[meta_index,'num']))
scaling_to_mm=real_r/r

l_list=[]
xlabel_list=[]
ylabel_list=[]
pos=pd.read_csv(fileplace+csv_name,header=0)
if predix=='1':
    pos=pos[pos['position']<=(10*60*fps)]
if predix=='2':
    pos=pos[(pos['position']<=(20*60*fps)) ]
if predix=='3':
    pos=pos[(pos['position']<=(30*60*fps)) ]
if predix=='4':
    pos=pos[(pos['position']<=(40*60*fps))]
if predix=='5':
    pos=pos[(pos['position']<=(50*60*fps))]

for i in range(n_inds):
    xlabel_list.append('x'+str(int(i)))
    ylabel_list.append('y'+str(int(i)))

for xlabel in xlabel_list:
    xlabel_p=xlabel+'_p'
    pos[xlabel]=pos[xlabel]*scaling
    pos[xlabel_p]=pos[xlabel].shift(1)
for ylabel in ylabel_list:
    ylabel_p=ylabel+'_p'
    pos[ylabel]=pos[ylabel]*scaling
    pos[ylabel_p]=pos[ylabel].shift(1)
pos=pos.fillna(method='bfill')
pos=pos.fillna(method='ffill')
def smooth(x, window_len=11, window='hanning'):
    # window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman' flat window will produce a moving average smoothing.
    s = np.r_[2 * x[0] - x[window_len - 1::-1], x,
              2 * x[-1] - x[-1:-window_len:-1]]
    if window == 'flat':  #moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len:-window_len + 1]
#
def dist_cost(x1,x2,x3,x4):
    return np.sqrt(np.power(x1-x2,2)+np.power(x3-x4,2))

def cal_angular(x,y,x1,y1,x2,y2):
    a=np.array([x-x1,y-y1])
    b=np.array([x1-x2,y1-y2])
    l_a=np.sqrt(a.dot(a))
    l_b=np.sqrt(b.dot(b))
    d_m=a.dot(b)
    cos_v=d_m/(l_a*l_b)
    ang=np.arccos(cos_v)
#弧度制
    return ang
    # return ang*180/np.pi
def roll_np(df: pd.DataFrame, apply_func: callable, window: int, return_col_num: int, **kwargs):
    """
    rolling with multiple columns on 2 dim pd.Dataframe
    * the result can apply the function which can return pd.Series with multiple columns

    call apply function with numpy ndarray
    :param return_col_num: 返回的列数
    :param apply_func:
    :param df:
    :param window
    :param kwargs:
    :return:
    """
    # move index to values
    v = df.values
    dim0, dim1 = v.shape
    stride0, stride1 = v.strides
    stride_values = stride(v, (dim0 - (window - 1), window, dim1), (stride0, stride0, stride1))
    result_values = np.full((dim0, return_col_num), np.nan)
    for idx, values in enumerate(stride_values, window - 1):
        # values : col 1 is index, other is value
        result_values[idx,] = apply_func(values, **kwargs)
    return result_values
def get_rvalue(narr, **kwargs):
    """
    :param narr:
    :return:
    """
    c = np.asarray(narr[:, 0],dtype=float)
    d = np.asarray(narr[:, 1],dtype=float)
    try:
        slope, intercept, _, _, _ = stats.linregress(x=c,y=d)
        yhat=slope*c+intercept
        r2=r2_score(d,yhat)
    except:
        r2=0
    return r2
ind_df=pd.DataFrame(columns={'videoname','id','xlabel','ylabel',
                             'search_area_time','search_area','search_area_unit_move',
                             'max_velocity','max_velocity_threshed',
                             'total_move_length','total_move_length_threshed',
                             'velocity','velocity_threshed',
                             'move_time','move_time_threshed',
                             'tracks_num','tracks_duration','tracks_length',
                             'long_stop_num','stop_duration','long_stop_duration',
                             'angular_velocity','angular_velocity_non_edge','angular_velocity_at_edge',
                             'track_straightness','track_straightness_non_edge','track_straightness_at_edge',
                             'max_angular_velocity','max_angular_velocity_non_edge','max_angular_velocity_at_edge',
                             'r_dist','r_edge','movelength_ratio_at_edge',
                             'move_proportion_at_edge','velocity_at_edge',
                             'move_proportion_at_center','velocity_at_center',
                             'meander','meander_non_edge','meander_at_edge',
                             'max_meander','max_meander_non_edge','max_meander_at_edge','max_velocity_at_edge','max_velocity_at_center','movelength_at_edge','movelength_at_center'
                             })
g_df=pd.DataFrame(columns={'videoname',
                             'search_area_time','search_area','search_area_unit_move',
                             'max_velocity','max_velocity_threshed',
                             'total_move_length','total_move_length_threshed',
                             'velocity','velocity_threshed',
                             'move_time','move_time_threshed',
                             'tracks_num','tracks_duration','tracks_length',
                             'long_stop_num','stop_duration','long_stop_duration',
                             'angular_velocity','angular_velocity_non_edge','angular_velocity_at_edge',
                             'track_straightness','track_straightness_non_edge','track_straightness_at_edge',
                             'max_angular_velocity','max_angular_velocity_non_edge','max_angular_velocity_at_edge',
                             'r_dist','r_edge','movelength_ratio_at_edge',
                             'move_proportion_at_edge','velocity_at_edge',
                             'move_proportion_at_center','velocity_at_center',
                             'meander','meander_non_edge','meander_at_edge',
                             'max_meander','max_meander_non_edge','max_meander_at_edge','max_velocity_at_edge','max_velocity_at_center','movelength_at_edge','movelength_at_center'
                             })

for i in range(n_inds):
    xlabel='x'+str(int(i))
    ylabel='y'+str(int(i))
    xlabel_p=xlabel+'_p'
    ylabel_p=ylabel+'_p'
    move_label='move'+str(int(i))
    move_label_bin='move_binary_'+str(int(i))
    long_no_move_label_bin='long_no_move_binary_'+str(int(i))
    ind_df.loc[i,'videoname']=v_name
    ind_df.loc[i,'id']=i
    ind_df.loc[i,'xlabel']=xlabel
    ind_df.loc[i,'ylabel']=ylabel

    pos[move_label]=pos.apply(lambda row: dist_cost(row[xlabel], row[xlabel_p], row[ylabel], row[ylabel_p]), axis=1)
    pos[move_label]=pos[move_label]*scaling_to_mm
    move_list=pos[move_label]
    total_move_length=np.sum(move_list)
    move_time=sum(move_list>0)/len(pos)
    max_velocity=np.percentile(move_list,100*max_v_thresh)*fps
    if sum(~np.isnan(move_list))>0:
        velocity=np.nanmean(move_list)*fps# in seconds
    else:
        velocity=np.nan

    pos[move_label_bin]=0
    pos.loc[pos[move_label]*fps>move_thresh,move_label_bin]=1
    true_move_list=move_list[move_list*fps>move_thresh]
    move_time_threshed=len(true_move_list)/len(pos)
    if move_time_threshed>0:
        max_velocity_threshed=np.percentile(true_move_list,100*max_v_thresh)*fps
        total_move_length_threshed=np.sum(true_move_list)
        velocity_threshed=np.nanmean(true_move_list)*fps
    else:
        max_velocity_threshed=np.nan
        total_move_length_threshed=np.nan
        velocity_threshed=np.nan

    ind_df.loc[i,'max_velocity']=max_velocity
    ind_df.loc[i,'total_move_length']=total_move_length
    ind_df.loc[i,'velocity']=velocity
    ind_df.loc[i,'move_time']=move_time
    ind_df.loc[i,'move_time_threshed']=move_time_threshed
    ind_df.loc[i,'max_velocity_threshed']=max_velocity_threshed
    ind_df.loc[i,'total_move_length_threshed']=total_move_length_threshed
    ind_df.loc[i,'velocity_threshed']=velocity_threshed

    r_label='r_dist'+str(int(i))
    r_edge='r_edge'+str(int(i))
    pos[r_label]=(np.sqrt(np.power(pos[xlabel]-x,2)+np.power(pos[ylabel]-y,2)))/r
    pos[r_edge]=0
    pos.loc[pos[r_label]>1-r_thresh,r_edge]=1
    ind_df.loc[i,'r_dist']=np.nanmean(pos[r_label])
    ind_df.loc[i,'r_edge']=np.nanmean(pos[r_edge])
    move_proportion_at_edge_list = pos.loc[pos[r_edge]==1,move_label_bin]
    if sum(~np.isnan(move_proportion_at_edge_list))>0:
        ind_df.loc[i,'move_proportion_at_edge']=np.nanmean(move_proportion_at_edge_list)
    else:
        ind_df.loc[i,'move_proportion_at_edge']=np.nan
    move_proportion_at_center_list=pos.loc[pos[r_edge]==0,move_label_bin]
    if sum(~np.isnan(move_proportion_at_center_list))>0:
        ind_df.loc[i,'move_proportion_at_center']=np.nanmean(move_proportion_at_center_list)
    else:
        ind_df.loc[i,'move_proportion_at_center']=np.nan
    velocity_at_edge_list=pos.loc[(pos[r_edge]==1)&(pos[move_label_bin]==1),move_label]
    if sum(~np.isnan(velocity_at_edge_list))>0:
        ind_df.loc[i,'velocity_at_edge']=np.nanmean(velocity_at_edge_list)*fps
        ind_df.loc[i,'max_velocity_at_edge']=np.percentile(velocity_at_edge_list,100*max_v_thresh)*fps
        ind_df.loc[i,'movelength_at_edge']=np.nansum(velocity_at_edge_list)
    else:
        ind_df.loc[i,'velocity_at_edge']=np.nan
        ind_df.loc[i,'max_velocity_at_edge']=np.nan
        ind_df.loc[i,'movelength_at_edge']=np.nan

    velocity_at_center_list=pos.loc[(pos[r_edge]==0)&(pos[move_label_bin]==1),move_label]
    if sum(~np.isnan(velocity_at_center_list))>0:
        ind_df.loc[i,'velocity_at_center']=np.nanmean(velocity_at_center_list)*fps
        ind_df.loc[i,'max_velocity_at_center']=np.percentile(velocity_at_center_list,100*max_v_thresh)*fps
        ind_df.loc[i,'movelength_at_center']=np.nansum(velocity_at_center_list)
    else:
        ind_df.loc[i,'velocity_at_center']=np.nan
        ind_df.loc[i,'max_velocity_at_center']=np.nan
        ind_df.loc[i,'movelength_at_center']=np.nan

    #rolling
    # track_straightness_window
    tmp_pos=pos[[xlabel,ylabel,'position',r_edge,move_label_bin]]
    tmp_pos['r2value']=roll_np(tmp_pos, get_rvalue, int(track_straightness_window*fps),1)
    tmp_pos.dropna(inplace=True)
    if len(tmp_pos)>0:
        track_straightness_list=tmp_pos.loc[tmp_pos[move_label_bin]==1,'r2value']
        if sum(~np.isnan(track_straightness_list))>0:
            ind_df.loc[i,'track_straightness']=np.nanmean(track_straightness_list)
        else:
            ind_df.loc[i,'track_straightness']=np.nan
        track_straightness_non_edge_list=tmp_pos.loc[(tmp_pos[move_label_bin]==1)&(tmp_pos[r_edge]==0),'r2value']
        if sum(~np.isnan(track_straightness_list))>0:
            ind_df.loc[i,'track_straightness_non_edge']=np.nanmean(track_straightness_list)
        else:
            ind_df.loc[i,'track_straightness_non_edge']=np.nan
        track_straightness_at_edge_list=tmp_pos.loc[(tmp_pos[move_label_bin]==1)&(tmp_pos[r_edge]==1),'r2value']
        if sum(~np.isnan(track_straightness_list))>0:
            ind_df.loc[i,'track_straightness_at_edge']=np.nanmean(track_straightness_at_edge_list)
        else:
            ind_df.loc[i,'track_straightness_at_edge']=np.nan
    else:
        ind_df.loc[i,'track_straightness']=np.nan
        ind_df.loc[i,'track_straightness_non_edge']=np.nan
        ind_df.loc[i,'track_straightness_at_edge']=np.nan
    print(v_name,i,'th dro. velocity and track_straightness finished')
    pos['move_label_bin']=pos[move_label_bin]
    pos['value_grp'] = (pos.move_label_bin.diff(1) != 0).astype('int').cumsum()
    pos['move']=pos[move_label]
    nomove_df=pd.DataFrame({'Begin' : pos.groupby('value_grp').position.first(),
                            'End' : pos.groupby('value_grp').position.last(),
                            'duration' : pos.groupby('value_grp').size(),
                            'length' : pos.groupby('value_grp').move.sum(),
                            'move' : pos.groupby('value_grp').move_label_bin.first()}).reset_index(drop=True)
    pos[long_no_move_label_bin]=0
    if len(nomove_df)>0:
        ind_df.loc[i,'tracks_num']=sum(nomove_df['move'])
        tracks_duration_list=nomove_df.loc[nomove_df['move']==1,'duration']
        if sum(~np.isnan(tracks_duration_list))>0:
            ind_df.loc[i,'tracks_duration']=(np.nanmean(tracks_duration_list))/fps# s
        else:
            ind_df.loc[i,'tracks_duration']=np.nan
        tracks_length_list=nomove_df.loc[nomove_df['move']==1,'length']
        if sum(~np.isnan(tracks_length_list))>0:
            ind_df.loc[i,'tracks_length']=(np.nanmean(tracks_length_list))# mm
        else:
            ind_df.loc[i,'tracks_length']=np.nan
        stop_duration_list=nomove_df.loc[nomove_df['move']==0,'duration']
        if sum(~np.isnan(stop_duration_list))>0:
            ind_df.loc[i,'stop_duration']=(np.nanmean(stop_duration_list))/fps# 秒为单位
        else:
            ind_df.loc[i,'stop_duration']=np.nan
        ind_df.loc[i,'long_stop_num']=len(nomove_df[(nomove_df['move']==0)&(nomove_df['duration']>(long_stop_thresh*fps))])
        long_stop_duration_list=nomove_df.loc[(nomove_df['move']==0)&(nomove_df['duration']>(long_stop_thresh*fps)),'duration']
        if sum(~np.isnan(long_stop_duration_list))>0:
            ind_df.loc[i,'long_stop_duration']=np.nanmean(long_stop_duration_list)/fps# 秒为单位
        else:
            ind_df.loc[i,'long_stop_duration']=np.nan
        long_no_move_df=nomove_df[(nomove_df['move']==0)&(nomove_df['duration']>(long_stop_thresh*fps))]
        long_no_move_df.reset_index(drop=True,inplace=True)
        for l in long_no_move_df.index:
            begin=long_no_move_df.loc[l,'Begin']
            end=long_no_move_df.loc[l,'End']
            pos.loc[begin:end,long_no_move_label_bin]=1
    else:
        ind_df.loc[i,'tracks_num']=np.nan
        ind_df.loc[i,'tracks_duration']=np.nan
        ind_df.loc[i,'tracks_length']=np.nan
        ind_df.loc[i,'stop_duration']=np.nan
        ind_df.loc[i,'long_stop_num']=np.nan
        ind_df.loc[i,'long_stop_duration']=np.nan
    #当前位置/windows前的位置/2*windows前的位置  0.2s
    pos['pos_x_previous1']=pos[xlabel].shift(int(angular_velocity_window*fps))
    pos['pos_x_previous2']=pos[xlabel].shift(int(2*angular_velocity_window*fps))
    pos['pos_y_previous1']=pos[ylabel].shift(int(angular_velocity_window*fps))
    pos['pos_y_previous2']=pos[ylabel].shift(int(2*angular_velocity_window*fps))
    #cal_angular(x,y,x1,y1,x2,y2)
    pos['angular']=pos.apply(lambda row: cal_angular(row[xlabel], row[ylabel], row['pos_x_previous1'], row['pos_y_previous1'], row['pos_x_previous2'], row['pos_y_previous2']), axis=1)
    pos['angular_dis']=pos.apply(lambda row: dist_cost(row[xlabel], row['pos_x_previous1'], row[ylabel], row['pos_y_previous1']), axis=1)
    pos['angular_dis']=pos['angular_dis']*scaling_to_mm
    pos['meander']=pos['angular']/pos['angular_dis']
    pos['angular']=pos['angular']/angular_velocity_window
    # pos['angular'].fillna(value=0,inplace=True)

    angular_list=pos.loc[(pos[move_label_bin]==1)&(~pos['angular'].isna()),'angular']
    if sum(~np.isnan(angular_list))>0:
        ind_df.loc[i,'max_angular_velocity']=np.percentile(angular_list,100*max_v_thresh)
        ind_df.loc[i,'angular_velocity']=np.nanmean(angular_list)
    else:
        ind_df.loc[i,'max_angular_velocity']=np.nan
        ind_df.loc[i,'angular_velocity']=np.nan
    angular_non_edge_list=pos.loc[(pos[move_label_bin]==1)&(pos[r_edge]==0)&(~pos['angular'].isna()),'angular']
    if sum(~np.isnan(angular_non_edge_list))>0:
        ind_df.loc[i,'max_angular_velocity_non_edge']=np.percentile(angular_non_edge_list,100*max_v_thresh)
        ind_df.loc[i,'angular_velocity_non_edge']=np.nanmean(angular_non_edge_list)
    else:
        ind_df.loc[i,'max_angular_velocity_non_edge']=np.nan
        ind_df.loc[i,'angular_velocity_non_edge']=np.nan
    angular_at_edge_list=pos.loc[(pos[move_label_bin]==1)&(pos[r_edge]==1)&(~pos['angular'].isna()),'angular']
    if sum(~np.isnan(angular_at_edge_list))>0:
        ind_df.loc[i,'max_angular_velocity_at_edge']=np.percentile(angular_at_edge_list,100*max_v_thresh)
        ind_df.loc[i,'angular_velocity_at_edge']=np.nanmean(angular_at_edge_list)
    else:
        ind_df.loc[i,'max_angular_velocity_at_edge']=np.nan
        ind_df.loc[i,'angular_velocity_at_edge']=np.nan

    meander_list=pos.loc[(pos[move_label_bin]==1)&(~pos['meander'].isna()),'meander']
    if sum(~np.isnan(meander_list))>0:
        ind_df.loc[i,'max_meander']=np.percentile(meander_list,100*max_v_thresh)
        ind_df.loc[i,'meander']=np.nanmedian(meander_list)
    else:
        ind_df.loc[i,'max_meander']=np.nan
        ind_df.loc[i,'meander']=np.nan
    meander_non_edge_list=pos.loc[(pos[move_label_bin]==1)&(pos[r_edge]==0)&(~pos['meander'].isna()),'meander']
    if sum(~np.isnan(meander_non_edge_list))>0:
        ind_df.loc[i,'max_meander_non_edge']=np.percentile(meander_non_edge_list,100*max_v_thresh)
        ind_df.loc[i,'meander_non_edge']=np.nanmedian(meander_non_edge_list)
    else:
        ind_df.loc[i,'max_meander_non_edge']=np.nan
        ind_df.loc[i,'meander_non_edge']=np.nan
    meander_at_edge_list=pos.loc[(pos[move_label_bin]==1)&(pos[r_edge]==1)&(~pos['meander'].isna()),'meander']
    if sum(~np.isnan(meander_at_edge_list))>0:
        ind_df.loc[i,'max_meander_at_edge']=np.percentile(meander_at_edge_list,100*max_v_thresh)
        ind_df.loc[i,'meander_at_edge']=np.nanmedian(meander_at_edge_list)
    else:
        ind_df.loc[i,'max_meander_at_edge']=np.nan
        ind_df.loc[i,'meander_at_edge']=np.nan
    print(v_name,i,'th dro. movetime, angular and meander finished')
# nomove_df.to_csv('1.csv')
    xlabel_int='x_i'
    ylabel_int='y_i'
    pos[xlabel_int]=pos[xlabel].astype(int)
    pos[ylabel_int]=pos[ylabel].astype(int)

    area=np.zeros(shape=[int(y_max),int(x_max)])
    mask = np.zeros(shape=[int(y_max),int(x_max)])
    cv2.circle(mask, (int(x),int(y)), int(r), 255, -1)
    ind_df.loc[i,'search_area_time']=1
    ind_df.loc[i,'search_area_unit_move']=0
    search_area=0
    # search_label='search'+str(int(i))
    for gi_ in pos.index:
        current_x = pos.loc[gi_, xlabel_int]
        current_y = pos.loc[gi_, ylabel_int]
        lim_x_min=int(max(current_x-sensing_area,x_min))
        lim_x_max = int(min(current_x + sensing_area, x_max))
        lim_y_min=int(max(current_y-sensing_area,y_min))
        lim_y_max = int(min(current_y + sensing_area, y_max))
        # 矩形
        area[lim_y_min:lim_y_max, lim_x_min:lim_x_max] = area[lim_y_min:lim_y_max, lim_x_min:lim_x_max] + 1
        area[mask < 250] = 0
        search_area=sum(sum(area>0.01))/cycle_area
        if search_area>area_thresh:
            ind_df.loc[i,'search_area_time']=gi_/len(pos)
            ind_df.loc[i,'search_area_unit_move']=search_area/sum(pos.loc[0:gi_,move_label])
            break
    if ind_df.loc[i,'search_area_time']==1:
        ind_df.loc[i,'search_area_unit_move']=search_area/sum(pos[move_label])
    tmp_pos=pos.loc[(pos['position']<=(area_search_time_thresh*60*fps)),[xlabel_int,ylabel_int,move_label]]
    gpos=pd.DataFrame({'count' : tmp_pos.groupby( [ xlabel_int, ylabel_int] ).size()}).reset_index()
    area=np.zeros(shape=[int(y_max),int(x_max)])
    for gi_ in gpos.index:
        current_x = gpos.loc[gi_, 'x_i']
        current_y = gpos.loc[gi_, 'y_i']
        count = gpos.loc[gi_, 'count']
        lim_x_min=int(max(current_x-sensing_area,x_min))
        lim_x_max = int(min(current_x + sensing_area, x_max))
        lim_y_min=int(max(current_y-sensing_area,y_min))
        lim_y_max = int(min(current_y + sensing_area, y_max))
        area[lim_y_min:lim_y_max, lim_x_min:lim_x_max] = area[lim_y_min:lim_y_max, lim_x_min:lim_x_max] + count

    area[mask < 250] = 0
    ind_df.loc[i,'search_area']=sum(sum(area>0.01))/cycle_area

    meander_label='meander'+str(int(i))
    angular_label='angular'+str(int(i))
    pos[meander_label]=pos['meander']
    pos[angular_label]=pos['angular']
    l_list.append(r_edge)
    l_list.append(move_label)
    l_list.append(move_label_bin)
    l_list.append(long_no_move_label_bin)
    l_list.append(meander_label)
    l_list.append(angular_label)
    print(v_name,i,'th dro. searchtime finished')
    #
    # plt.figure(figsize=(11, 8),facecolor='lightgray')
    # plt.title(xlabel+ylabel+' motion region')
    # plt.grid(linestyle=":")
    # plt.imshow(area, cmap='jet')
    # plt.colorbar()
    # plt.xlim(0, x_max)
    # plt.ylim(0, y_max)
    # plt.gca().invert_yaxis()
    # plt.savefig(fileplace_analysis + v_name+'_'+xlabel+'_'+ylabel + '_motion_region.png')
    # plt.close()

    plt.figure(figsize=(11,8))
    plt.plot(pos[xlabel], pos[ylabel], alpha=0.5)
    plt.xlabel('X', fontsize=10)
    plt.ylabel('Y', fontsize=10)
    plt.xlim(0, x_max)
    plt.ylim(0, y_max)
    plt.gca().invert_yaxis()
    plt.title(xlabel+ylabel+' motion trace')
    plt.tight_layout()
    plt.savefig(fileplace_analysis + v_name+'_'+xlabel+'_'+ylabel + '_motion_trace.png')
    plt.close()

    pos[move_label]=pos[move_label]*fps
    smoothed_velocity_list=smooth(np.array(list(pos[move_label])),window_len=fps*60)
    frame_list=pos['position']/fps/60 # in minute
    plt.figure(figsize=(18,6))
    plt.plot(frame_list, pos[move_label], alpha=0.5,color='blue',label='velocity')
    plt.plot(frame_list, smoothed_velocity_list, alpha=0.5,color='red',label='smoothed velocity')
    plt.xlabel('Time(min)', fontsize=10)
    plt.ylabel('Velocity(mm/sec)', fontsize=10)
    # ax1.set_xlim(min_frame, max_frame)
    # plt.ylim(0, y_max)
    plt.title(xlabel+ylabel+' motion velocity')
    plt.tight_layout()
    plt.savefig(fileplace_analysis + v_name+'_'+xlabel+'_'+ylabel + '_motion_velocity.png')
    plt.close()

#'velocity','max_velocity','total_move_length','move_time',

ind_df['movelength_ratio_at_edge']=ind_df['movelength_at_edge']/ind_df['total_move_length_threshed']

ind_df_tocsv=ind_df[['videoname','id','xlabel','ylabel',
               'search_area_time','search_area','search_area_unit_move',
               'velocity_threshed','velocity_at_edge','velocity_at_center',
               'max_velocity_threshed','max_velocity_at_edge','max_velocity_at_center',
               'total_move_length_threshed','movelength_at_edge','movelength_at_center',
               'move_time_threshed','move_proportion_at_edge','move_proportion_at_center',
               'r_dist','r_edge','movelength_ratio_at_edge',
               'tracks_num','tracks_duration','tracks_length',
               'long_stop_num','stop_duration','long_stop_duration',
               'track_straightness','track_straightness_non_edge','track_straightness_at_edge',
               'angular_velocity','angular_velocity_non_edge','angular_velocity_at_edge',
               'max_angular_velocity','max_angular_velocity_non_edge','max_angular_velocity_at_edge',
                'meander','meander_non_edge','meander_at_edge',
               'max_meander','max_meander_non_edge','max_meander_at_edge'
               ]]
ind_df_tocsv.to_csv(fileplace_analysis + v_name + '_ind_motion.csv')
g_df.loc[0,'videoname']=v_name

g_df['movelength_ratio_at_edge']=np.nanmean(list(ind_df['movelength_ratio_at_edge']))
g_df['search_area_time']=np.nanmean(list(ind_df['search_area_time']))
g_df['search_area']=np.nanmean(list(ind_df['search_area']))
g_df['search_area_unit_move']=np.nanmean(list(ind_df['search_area_unit_move']))
g_df['max_velocity']=np.nanmean(list(ind_df['max_velocity']))
g_df['max_velocity_threshed']=np.nanmean(list(ind_df['max_velocity_threshed']))
g_df['total_move_length']=np.nanmean(list(ind_df['total_move_length']))
g_df['total_move_length_threshed']=np.nanmean(list(ind_df['total_move_length_threshed']))
g_df['velocity']=np.nanmean(list(ind_df['velocity']))
g_df['velocity_threshed']=np.nanmean(list(ind_df['velocity_threshed']))
g_df['velocity_at_edge']=np.nanmean(list(ind_df['velocity_at_edge']))
g_df['velocity_at_center']=np.nanmean(list(ind_df['velocity_at_center']))
g_df['move_time']=np.nanmean(list(ind_df['move_time']))
g_df['move_time_threshed']=np.nanmean(list(ind_df['move_time_threshed']))
g_df['move_proportion_at_edge']=np.nanmean(list(ind_df['move_proportion_at_edge']))
g_df['move_proportion_at_center']=np.nanmean(list(ind_df['move_proportion_at_center']))
g_df['r_dist']=np.nanmean(list(ind_df['r_dist']))
g_df['r_edge']=np.nanmean(list(ind_df['r_edge']))
g_df['tracks_num']=np.nanmean(list(ind_df['tracks_num']))
g_df['tracks_duration']=np.nanmean(list(ind_df['tracks_duration']))
g_df['tracks_length']=np.nanmean(list(ind_df['tracks_length']))
g_df['long_stop_num']=np.nanmean(list(ind_df['long_stop_num']))
g_df['stop_duration']=np.nanmean(list(ind_df['stop_duration']))
g_df['long_stop_duration']=np.nanmean(list(ind_df['long_stop_duration']))
g_df['angular_velocity']=np.nanmean(list(ind_df['angular_velocity']))
g_df['angular_velocity_non_edge']=np.nanmean(list(ind_df['angular_velocity_non_edge']))
g_df['angular_velocity_at_edge']=np.nanmean(list(ind_df['angular_velocity_at_edge']))
g_df['track_straightness']=np.nanmean(list(ind_df['track_straightness']))
g_df['track_straightness_non_edge']=np.nanmean(list(ind_df['track_straightness_non_edge']))
g_df['track_straightness_at_edge']=np.nanmean(list(ind_df['track_straightness_at_edge']))
g_df['max_angular_velocity']=np.nanmean(list(ind_df['max_angular_velocity']))
g_df['max_angular_velocity_non_edge']=np.nanmean(list(ind_df['max_angular_velocity_non_edge']))
g_df['max_angular_velocity_at_edge']=np.nanmean(list(ind_df['max_angular_velocity_at_edge']))
g_df['meander']=np.nanmean(list(ind_df['meander']))
g_df['meander_non_edge']=np.nanmean(list(ind_df['meander_non_edge']))
g_df['meander_at_edge']=np.nanmean(list(ind_df['meander_at_edge']))
g_df['max_meander']=np.nanmean(list(ind_df['max_meander']))
g_df['max_meander_non_edge']=np.nanmean(list(ind_df['max_meander_non_edge']))
g_df['max_meander_at_edge']=np.nanmean(list(ind_df['max_meander_at_edge']))

g_df['movelength_at_edge']=np.nanmean(list(ind_df['movelength_at_edge']))
g_df['movelength_at_center']=np.nanmean(list(ind_df['movelength_at_center']))
g_df['max_velocity_at_edge']=np.nanmean(list(ind_df['max_velocity_at_edge']))
g_df['max_velocity_at_center']=np.nanmean(list(ind_df['max_velocity_at_center']))


#'max_velocity','velocity','total_move_length','move_time',
g_df_tocsv=g_df[['videoname',
               'search_area_time','search_area','search_area_unit_move',
               'velocity_threshed','velocity_at_edge','velocity_at_center',
               'max_velocity_threshed','max_velocity_at_edge','max_velocity_at_center',
               'total_move_length_threshed','movelength_at_edge','movelength_at_center',
               'move_time_threshed','move_proportion_at_edge','move_proportion_at_center',
               'r_dist','r_edge','movelength_ratio_at_edge',
               'tracks_num','tracks_duration','tracks_length',
               'long_stop_num','stop_duration','long_stop_duration',
               'track_straightness','track_straightness_non_edge','track_straightness_at_edge',
                'angular_velocity','angular_velocity_non_edge','angular_velocity_at_edge',
               'max_angular_velocity','max_angular_velocity_non_edge','max_angular_velocity_at_edge',
               'meander','meander_non_edge','meander_at_edge',
               'max_meander','max_meander_non_edge','max_meander_at_edge'
               ]]
g_df_tocsv.to_csv(fileplace_analysis + v_name + '_avg_motion.csv')

moveinfo=pos[l_list+['position']]
moveinfo.to_csv(fileplace_analysis + v_name + '_ind_moveinfo.csv',index=False)

plt.figure(figsize=(11,8))
import random

for i in range(n_inds):
    xlabel='x'+str(int(i))
    ylabel='y'+str(int(i))
    plt.plot(pos[xlabel], pos[ylabel],color=(random.random(),random.random(),random.random()), alpha=0.5)
plt.xlabel('X', fontsize=10)
plt.ylabel('Y', fontsize=10)
plt.xlim(0, x_max)
plt.ylim(0, y_max)
plt.gca().invert_yaxis()
plt.title(v_name+' motion trace')
plt.tight_layout()
plt.savefig(fileplace_analysis + v_name+ '_motion_trace.png')
plt.close()

move_label_list=[]
for i in range(n_inds):
    move_label='move'+str(int(i))
    move_label_list.append(move_label)
pos['avg_move']=pos[move_label_list].mean(axis=1)
pos['avg_move']=pos['avg_move']
smoothed_velocity_list=smooth(np.array(list(pos['avg_move'])),window_len=fps*10)
frame_list=pos['position']/fps/60 # in minute
plt.figure(figsize=(18,6))
plt.plot(frame_list, pos['avg_move'], alpha=0.5,color='blue',label='velocity')
plt.plot(frame_list, smoothed_velocity_list, alpha=0.5,color='red',label='smoothed velocity')
plt.xlabel('Time(min)', fontsize=10)
plt.ylabel('Velocity(mm/sec)', fontsize=10)
# ax1.set_xlim(min_frame, max_frame)
# plt.ylim(0, y_max)
plt.title(v_name+' motion velocity')
plt.tight_layout()
plt.savefig(fileplace_analysis + v_name + '_motion_velocity.png')
plt.close()