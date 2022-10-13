# -*- coding: utf-8 -*-
from numpy.lib.stride_tricks import as_strided as stride
from sklearn.metrics import r2_score
from scipy import stats
import numpy as np
from vt_gui import Ui_Videotrack
from PyQt5.QtCore import QDir,Qt,QCoreApplication
import networkx as nx
from matplotlib import cm
from matplotlib.colors import ListedColormap
# from moviepy.editor import *
from PyQt5.QtWidgets import  QFileDialog, QMainWindow,QApplication,QMessageBox
from PyQt5.QtGui import QIntValidator,QDoubleValidator
import os
import pandas as pd
import cv2
import platform
from PyQt5.QtGui import QImage, QPixmap
import matplotlib.pyplot as plt
import sys
import math
import random
# path = getattr(sys, '_MEIPASS', os.getcwd())
# os.chdir(path)

s = platform.uname()
os_p = s[0]


def annoateVideo(saveplace,tracefile,videoname,n_inds,predix,scale):
    colorlist = [(60, 0, 255),
                 (140, 0, 255),
                 (220, 0, 255),
                 (255, 0, 200),
                 (255, 0, 120),
                 (255, 0, 40),
                 (255, 20, 0),
                 (255, 100, 0),
                 (255, 180, 0),
                 (240, 255, 0),
                 (160, 255, 0),
                 (80, 255, 0),
                 (0, 255, 0),
                 (0, 255, 60),
                 (0, 255, 140),
                 (0, 255, 220)]
    video=videoname
    outvideo=saveplace+'/'+predix+'annoated_video.mp4'
    trace=tracefile
    trace=pd.read_csv(trace,header=0,index_col=False)
    cap = cv2.VideoCapture(video)
    n_inds = int(n_inds)
    subject_list= ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H','I','J','K','L','M','N']
    xlabel_list=[]
    ylabel_list=[]
    for i in range(n_inds):
        xlabel_list.append('x'+str(int(i)))
        ylabel_list.append('y'+str(int(i)))
    codec = 'mp4v'
    fps=cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    x_dim=int(cap.read()[1].shape[0])
    y_dim=int(cap.read()[1].shape[1])
    font = cv2.FONT_HERSHEY_SIMPLEX

    output_framesize = (y_dim,x_dim)
    red = (48, 48, 255)
    green = (34, 139, 34)
    yellow = (0, 255, 255)
    out = cv2.VideoWriter(filename=outvideo, fourcc=fourcc, fps=fps, frameSize=output_framesize, isColor=True)
    last=0
    while True:
        res, image = cap.read()
        this = int(cap.get(1))
        if res == True:
            # tmp_trace = trace[trace['position'] == this]
            # tmp_trace = tmp_trace.reset_index(drop=True)
            # if len(tmp_trace)==0:
            #     continue
            # for i in range(n_inds):
            #     x_label=xlabel_list[i]
            #     y_label = ylabel_list[i]
            #     x = int(tmp_trace.loc[0,x_label]*scale)
            #     y = int(tmp_trace.loc[0, y_label]*scale)
            #     cv2.circle(image, (x,y), 3, yellow, -1, cv2.LINE_AA)
            #     cv2.putText(image, subject_list[i], (x+5,y+15), font, 0.5, green, 2)
            # cv2.putText(image, str(int(this)), (5, 30), font, 1, (255, 255, 255), 2)
            # out.write(image)
            # this=int(this)
            this_list = np.array(range(max(0, this - 15), this + 1))
            tmp_trace = trace[trace['position'].isin(this_list)]
            tmp_trace = tmp_trace.reset_index(drop=True)
            if len(tmp_trace) == 0:
                continue
            for i in range(n_inds):
                x_label = xlabel_list[i]
                y_label = ylabel_list[i]
                for j in tmp_trace.index:
                    x = int(tmp_trace.loc[j, x_label] * scale)
                    y = int(tmp_trace.loc[j, y_label] * scale)
                    cv2.circle(image, (x, y), 2, colorlist[j], -1, cv2.LINE_AA)
                cv2.putText(image, subject_list[i], (x + 5, y + 15), font, 0.5, green, 1)
            cv2.putText(image, str(int(this)), (5, 30), font, 1, (255, 255, 255), 2)
            out.write(image)

        if last >= this:
            break
        last = this
        if this%180==0:
            print(this/30)
    print('Traced video generation finished')
    cap.release()
    out.release()
    return None


def group_interaction_analysis(saveplace,tracefile,videoname,x,y,r,n_inds,predix,real_r,sensing_area,scale_to_interaction,ssi_bin,fps,network_size,scaling):
    fileplace_analysis = saveplace + '/'
    network_size=math.ceil(n_inds*(n_inds-1)/2*network_size)
    csv_name =tracefile
    v_name = videoname

    if n_inds > 1:
        # scaling = 1.6  # 1.6 for 480p to 768p.  real_r/r for pixel to mm
        scaling_to_mm = real_r / r
        xlabel_list = []
        ylabel_list = []
        pos = pd.read_csv(csv_name, header=0, index_col=None)
        moveinfo = pd.read_csv(fileplace_analysis + predix + '_ind_moveinfo.csv', header=0, index_col=None)
        l_list = list(moveinfo.columns)
        pos = pd.merge(pos, moveinfo, on='position', how='inner')

        for i in range(n_inds):
            xlabel_list.append('x' + str(int(i)))
            ylabel_list.append('y' + str(int(i)))
        for xlabel in xlabel_list:
            pos[xlabel] = pos[xlabel] * scaling
        for ylabel in ylabel_list:
            pos[ylabel] = pos[ylabel] * scaling

        for i in range(n_inds):
            for j in range(n_inds):
                if j <= i:
                    continue
                xlab_i = 'x' + str(int(i))
                xlab_j = 'x' + str(int(j))
                ylab_i = 'y' + str(int(i))
                ylab_j = 'y' + str(int(j))
                dis_ij = 'dis_' + str(int(i)) + '_' + str(int(j))
                pos[dis_ij] = np.sqrt(np.power(pos[xlab_j] - pos[xlab_i], 2) + np.power(pos[ylab_j] - pos[ylab_i], 2))
                dis_ij_interaction = str(int(i)) + '_' + str(int(j)) + '_' + 'interaction'
                pos[dis_ij_interaction] = 0
                pos.loc[pos[dis_ij] < scale_to_interaction * sensing_area, dis_ij_interaction] = 1
                dis_ji = 'dis_' + str(int(j)) + '_' + str(int(i))
                dis_ji_interaction = str(int(j)) + '_' + str(int(i)) + '_' + 'interaction'
                pos[dis_ji_interaction] = pos[dis_ij_interaction]
                pos[dis_ji] = pos[dis_ij]
                l_list.append(dis_ij_interaction)
                l_list.append(dis_ji_interaction)

        columns = pos.columns
        g_df = pd.DataFrame(columns={'videoname',
                                     'acquaintances',
                                     'distance_space',
                                     'distance_space_at_edge',
                                     'distance_space_at_center',
                                     'distance_space_at_move',
                                     'distance_space_at_stop',
                                     'SSI',
                                     'SSI_at_edge',
                                     'SSI_at_center',
                                     'SSI_at_move',
                                     'SSI_at_stop',
                                     'interaction',
                                     'interaction_at_edge',
                                     'interaction_at_center',
                                     'interaction_at_edge_proportion',
                                     'interaction_at_center_proportion',
                                     'interaction_at_move',
                                     'interaction_at_stop',
                                     'interaction_at_move_proportion',
                                     'interaction_at_stop_proportion',
                                     'interaction_at_long_stop',
                                     'interaction_at_long_stop_proportion',
                                     'interaction_counts',
                                     'interaction_duration',
                                     'interaction_members',
                                     'degree_assortativity_coefficient',
                                     'clustering_coefficient',
                                     'betweenness_centrality',
                                     'diameter',
                                     'degree',
                                     'unconnected_social_network_proportion','global_efficiency'})
        g_df.loc[0, 'videoname'] = v_name
        g_df.loc[0, 'degree_assortativity_coefficient'] = []
        g_df.loc[0, 'clustering_coefficient'] = []
        g_df.loc[0, 'betweenness_centrality'] = 0
        g_df.loc[0, 'diameter'] = []
        # g_df.loc[0, 'sigma'] = []
        # g_df.loc[0, 'omega'] = []
        g_df.loc[0, 'degree'] = []
        g_df.loc[0, 'global_efficiency'] = []

        ind_df = pd.DataFrame(columns={'videoname', 'id', 'xlabel', 'ylabel',
                                       'acquaintances',
                                       'distance_space',
                                       'distance_space_at_edge',
                                       'distance_space_at_center',
                                       'distance_space_at_move',
                                       'distance_space_at_stop',
                                       'SSI',
                                       'SSI_at_edge',
                                       'SSI_at_center',
                                       'SSI_at_move',
                                       'SSI_at_stop',
                                       'interaction',
                                       'interaction_at_edge',
                                       'interaction_at_center',
                                       'interaction_at_edge_proportion',
                                       'interaction_at_center_proportion',
                                       'interaction_at_move',
                                       'interaction_at_stop',
                                       'interaction_at_move_proportion',
                                       'interaction_at_stop_proportion',
                                       'interaction_at_long_stop',
                                       'interaction_at_long_stop_proportion',
                                       'interaction_counts',
                                       'interaction_duration',
                                       'interaction_members',
                                       'betweenness_centrality',
                                       'degree',
                                       'clustering_coefficient',
                                       'closeness_centrality',
                                       'eccentricity',
                                       'dominating'
                                       })

        interaction_df = pd.DataFrame(
            columns={'videoname', 'id1', 'xlabel1', 'ylabel1', 'id2', 'xlabel2', 'ylabel2', 'interaction'})

        net_list = pd.DataFrame(columns={'subject1', 'subject2', 'label'})
        net_col = []
        count = 0
        for i in range(n_inds):
            for j in range(n_inds):
                if j <= i:
                    continue
                dis_ij_interaction = str(int(i)) + '_' + str(int(j)) + '_' + 'interaction'
                net_list.loc[count, 'subject1'] = i
                net_list.loc[count, 'subject2'] = j
                net_list.loc[count, 'label'] = dis_ij_interaction
                net_col.append(dis_ij_interaction)
                count = count + 1
        total_net = pos[net_col + ['position']]

        total_grouped = pd.DataFrame()
        for i in range(n_inds):
            for j in range(n_inds):
                if j <= i:
                    continue
                dis_ij_interaction = str(int(i)) + '_' + str(int(j)) + '_' + 'interaction'
                sub_net = pos[[dis_ij_interaction, 'position']]
                sub_net['interaction'] = sub_net[dis_ij_interaction]
                sub_net['value_grp'] = (sub_net.interaction.diff(1) != 0).astype('int').cumsum()
                grouped_sub_net = pd.DataFrame({'Begin': sub_net.groupby('value_grp').position.first(),
                                                'End': sub_net.groupby('value_grp').position.last(),
                                                'time_duration': sub_net.groupby('value_grp').size(),
                                                'interactions': sub_net.groupby(
                                                    'value_grp').interaction.first()}).reset_index(drop=True)
                grouped_sub_net = grouped_sub_net[grouped_sub_net['interactions'] == 1]
                grouped_sub_net['subject1'] = i
                grouped_sub_net['subject2'] = j
                if len(total_grouped) > 0:
                    total_grouped = pd.concat([total_grouped, grouped_sub_net])
                if len(total_grouped) == 0:
                    total_grouped = grouped_sub_net
        total_grouped = total_grouped.sort_values(by='End')
        total_grouped.reset_index(drop=True, inplace=True)
        total_grouped.to_csv(fileplace_analysis + predix + '_interaction_bytimeline.csv')
        count = 0
        for s in range(n_inds):
            ind_df.loc[s, 'betweenness_centrality'] = []
            ind_df.loc[s, 'clustering_coefficient'] = []
            ind_df.loc[s, 'degree'] = []
            ind_df.loc[s, 'closeness_centrality'] = []
            ind_df.loc[s, 'eccentricity'] = []
            ind_df.loc[s, 'dominating'] = 0
        er = 0
        er2=0
        for i in range(int(len(total_grouped) / network_size)):
            start = i * network_size
            end = (i + 1) * network_size
            if end <= len(total_grouped):
                subnet = total_grouped[start:end]
                G = nx.Graph()
                for j in subnet.index:
                    if (subnet.loc[j, 'subject1'], subnet.loc[j, 'subject2']) in G.edges():
                        G[subnet.loc[j, 'subject1']][subnet.loc[j, 'subject2']]['weight'] = \
                            G[subnet.loc[j, 'subject1']][subnet.loc[j, 'subject2']]['weight'] + subnet.loc[j, 'time_duration']
                    else:
                        G.add_weighted_edges_from([(subnet.loc[j, 'subject1'],
                                                    subnet.loc[j, 'subject2'],
                                                    subnet.loc[j, 'time_duration'])])
                if nx.is_connected(G):
                    pass
                else:
                    er = er + 1
                try:
                    bc = nx.betweenness_centrality(G)
                    ce = nx.clustering(G)
                    cc = nx.closeness_centrality(G)
                    ds = nx.dominating_set(G)
                    ec = nx.eccentricity(G)
                    dac = nx.degree_assortativity_coefficient(G)
                    acc = nx.average_clustering(G)
                    dg = nx.diameter(G)
                    ge = nx.global_efficiency(G)
                    obj_list = list(set(list(subnet['subject1']) + list(subnet['subject2'])))
                    for s in obj_list:
                        if G.degree(s) is not None:
                            ind_df.loc[s, 'degree'].append(G.degree(s))
                        if bc.get(s) is not None:
                            ind_df.loc[s, 'betweenness_centrality'].append(bc.get(s))
                        if ce.get(s) is not None:
                            ind_df.loc[s, 'clustering_coefficient'].append(ce.get(s))
                        if cc.get(s) is not None:
                            ind_df.loc[s, 'closeness_centrality'].append(cc.get(s))
                        if ec.get(s) is not None:
                            ind_df.loc[s, 'eccentricity'].append(ec.get(s))
                        if s in ds:
                            ind_df.loc[s, 'dominating'] = ind_df.loc[s, 'dominating'] + 1
                    if ge is not None and (not np.isnan(ge)):
                        g_df.loc[0, 'global_efficiency'].append(ge)
                    if dac is not None and (not np.isnan(dac)):
                        g_df.loc[0, 'degree_assortativity_coefficient'].append(dac)

                    if acc is not None and (not np.isnan(acc)):
                        g_df.loc[0, 'clustering_coefficient'].append(acc)
                    if dg is not None and (not np.isnan(dg)):
                        g_df.loc[0, 'diameter'].append(dg)

                    count = count + 1
                except:
                    er2=er2+1
                    pass

            # print(v_name, i, 'th dro. social network topological analysis finished. Total: ',
            #       str(int(len(total_grouped) / network_size)))
        print(v_name, ' dro. social network topological analysis finished. Total: ',
              str(int(len(total_grouped) / network_size)),' Error: ',er2)
        # print(er,'error',count,'all')
        for s in range(n_inds):
            if len(ind_df.loc[s, 'betweenness_centrality'])>0:
                ind_df.loc[s, 'betweenness_centrality'] = np.nanmean(ind_df.loc[s, 'betweenness_centrality'])
            else:
                ind_df.loc[s, 'betweenness_centrality'] =np.nan
            if len(ind_df.loc[s, 'clustering_coefficient'])>0:
                ind_df.loc[s, 'clustering_coefficient'] = np.nanmean(ind_df.loc[s, 'clustering_coefficient'])
            else:
                ind_df.loc[s, 'clustering_coefficient'] =np.nan
            if len(ind_df.loc[s, 'degree'])>0:
                ind_df.loc[s, 'degree'] = np.nanmean(ind_df.loc[s, 'degree'])
            else:
                ind_df.loc[s, 'degree'] =np.nan
            if len(ind_df.loc[s, 'closeness_centrality'])>0:
                ind_df.loc[s, 'closeness_centrality'] = np.nanmean(ind_df.loc[s, 'closeness_centrality'])
            else:
                ind_df.loc[s, 'closeness_centrality']=np.nan
            if len(ind_df.loc[s, 'eccentricity'])>0:
                ind_df.loc[s, 'eccentricity'] = np.nanmean(ind_df.loc[s, 'eccentricity'])
            else:
                ind_df.loc[s, 'eccentricity']=np.nan
            if count>0:
                ind_df.loc[s, 'dominating'] = ind_df.loc[s, 'dominating'] / count
            else:
                ind_df.loc[s, 'dominating'] =np.nan
        if sum(~np.isnan(list(ind_df['degree'])))>0:
            g_df.loc[0, 'degree'] = np.nanmean(ind_df['degree'])
        else:
            g_df.loc[0, 'degree'] =np.nan
        if sum(~np.isnan(list(ind_df['betweenness_centrality'])))>0:
            g_df.loc[0, 'betweenness_centrality'] = np.nanmean(ind_df['betweenness_centrality'])
        else:
            g_df.loc[0, 'betweenness_centrality'] =np.nan
        if len(g_df.loc[0, 'degree_assortativity_coefficient'])>0:
            g_df.loc[0, 'degree_assortativity_coefficient'] = np.nanmean(g_df.loc[0, 'degree_assortativity_coefficient'])
        else:
            g_df.loc[0, 'degree_assortativity_coefficient'] =np.nan
        if len(g_df.loc[0, 'clustering_coefficient'])>0:
            g_df.loc[0, 'clustering_coefficient'] = np.nanmean(g_df.loc[0, 'clustering_coefficient'])
        else:
            g_df.loc[0, 'clustering_coefficient']=np.nan
        if len(g_df.loc[0, 'diameter'])>0:
            g_df.loc[0, 'diameter'] = np.nanmean(g_df.loc[0, 'diameter'])
        else:
            g_df.loc[0, 'diameter'] =np.nan
        if len(g_df.loc[0, 'global_efficiency'])>0:
            g_df.loc[0, 'global_efficiency'] = np.nanmean(g_df.loc[0, 'global_efficiency'])
        else:
            g_df.loc[0, 'global_efficiency'] =np.nan


        # if len(g_df.loc[0, 'sigma'])>0:
        #     g_df.loc[0, 'sigma'] = np.nanmean(g_df.loc[0, 'sigma'])
        # else:
        #     g_df.loc[0, 'sigma'] =np.nan
        # if len(g_df.loc[0, 'omega']):
        #     g_df.loc[0, 'omega'] = np.nanmean(g_df.loc[0, 'omega'])
        # else:
        #     g_df.loc[0, 'omega'] =np.nan
        if er+count>0:
            g_df.loc[0, 'unconnected_social_network_proportion'] = er / (er + count)
        else:
            g_df.loc[0, 'unconnected_social_network_proportion'] = np.nan
        count = 0
        for i in range(n_inds):
            xlabel = 'x' + str(int(i))
            ylabel = 'y' + str(int(i))
            r_edge = 'r_edge' + str(int(i))
            move_label_bin = 'move_binary_' + str(int(i))
            long_no_move_label_bin = 'long_no_move_binary_' + str(int(i))
            col_list = []
            for col in columns:
                if 'dis_' + str(int(i)) in col:
                    col_list.append(col)
            tmp_dis = pos[col_list]
            tmp_dis['min_idx'] = tmp_dis.idxmin(axis=1)
            tmp_dis['min_val'] = tmp_dis.min(axis=1)
            tmp_dis['position'] = pos['position']
            ind_df.loc[i, 'videoname'] = v_name
            ind_df.loc[i, 'id'] = i
            ind_df.loc[i, 'xlabel'] = xlabel
            ind_df.loc[i, 'ylabel'] = ylabel
            ind_df.loc[i, 'distance_space'] = np.nanmean(tmp_dis['min_val']) * scaling_to_mm
            dis_label = str(int(i)) + '_dis_space'
            pos[dis_label] = tmp_dis['min_val'] * scaling_to_mm
            l_list.append(dis_label)
            ind_df.loc[i, 'SSI'] =(len(pos[pos[dis_label] <= ssi_bin]) - len(
                pos[(pos[dis_label] > ssi_bin) & (pos[dis_label] <= 2 * ssi_bin)])) / len(pos)
            if sum(~np.isnan(pos.loc[pos[r_edge] == 1, dis_label]))>0:
                ind_df.loc[i, 'distance_space_at_edge'] = np.nanmean(pos.loc[pos[r_edge] == 1, dis_label])
            else:
                ind_df.loc[i, 'distance_space_at_edge']=np.nan
            if sum(~np.isnan(pos.loc[pos[r_edge] == 0, dis_label]))>0:
                ind_df.loc[i, 'distance_space_at_center'] = np.nanmean(pos.loc[pos[r_edge] == 0, dis_label])
            else:
                ind_df.loc[i, 'distance_space_at_center']=np.nan
            if len(pos[pos[r_edge] == 1])>0:
                ind_df.loc[i, 'SSI_at_edge'] = (len(pos[(pos[dis_label] < ssi_bin) & (pos[r_edge] == 1)]) - len(
                    pos[(pos[dis_label] > ssi_bin) & (pos[dis_label] <= 2 * ssi_bin) & (pos[r_edge] == 1)])) / len(
                    pos[pos[r_edge] == 1])
            else:
                ind_df.loc[i, 'SSI_at_edge'] =np.nan
            if len(pos[pos[r_edge] == 0])>0:
                ind_df.loc[i, 'SSI_at_center'] = (len(pos[(pos[dis_label] < ssi_bin) & (pos[r_edge] == 0)]) - len(
                    pos[(pos[dis_label] > ssi_bin) & (pos[dis_label] <= 2 * ssi_bin) & (pos[r_edge] == 0)])) / len(
                    pos[pos[r_edge] == 0])
            else:
                ind_df.loc[i, 'SSI_at_center']=np.nan
            if sum(~np.isnan(pos.loc[pos[move_label_bin] == 1, dis_label]))>0:
                ind_df.loc[i, 'distance_space_at_move'] = np.nanmean(pos.loc[pos[move_label_bin] == 1, dis_label])
            else:
                ind_df.loc[i, 'distance_space_at_move']=np.nan
            if sum(~np.isnan(pos.loc[pos[move_label_bin] == 0, dis_label]))>0:
                ind_df.loc[i, 'distance_space_at_stop'] = np.nanmean(pos.loc[pos[move_label_bin] == 0, dis_label])
            else:
                ind_df.loc[i, 'distance_space_at_stop'] =np.nan
            if  len(pos[pos[move_label_bin] == 1])>0:
                ind_df.loc[i, 'SSI_at_move'] = (len(pos[(pos[dis_label] < ssi_bin) & (pos[move_label_bin] == 1)]) - len(
                    pos[(pos[dis_label] > ssi_bin) & (pos[dis_label] <= 2 * ssi_bin) & (pos[move_label_bin] == 1)])) / len(
                    pos[pos[move_label_bin] == 1])
            else:
                ind_df.loc[i, 'SSI_at_move'] =np.nan
            if  len(pos[pos[move_label_bin] == 0])>0:
                ind_df.loc[i, 'SSI_at_stop'] = (len(pos[(pos[dis_label] < ssi_bin) & (pos[move_label_bin] == 0)]) - len(
                    pos[(pos[dis_label] > ssi_bin) & (pos[dis_label] <= 2 * ssi_bin) & (pos[move_label_bin] == 0)])) / len(
                    pos[pos[move_label_bin] == 0])
            else:
                ind_df.loc[i, 'SSI_at_stop']=np.nan
            print(v_name, i, 'th dro. distance space analysis finished')
            frame_list = tmp_dis['position'] / fps / 60  # in minute
            distance_space = tmp_dis['min_val'] * scaling_to_mm
            plt.figure(figsize=(18, 6))
            plt.plot(frame_list, distance_space, alpha=0.5)
            plt.xlabel('Time(min)', fontsize=10)
            plt.ylabel('Distance(mm)', fontsize=10)
            plt.title(xlabel + ylabel + ' Distance from other drosophila')
            plt.tight_layout()
            plt.savefig(fileplace_analysis + predix + '_' + xlabel + '_' + ylabel + '_social_space.png')
            plt.close()

            col_list = []
            for col in columns:
                if str(int(i)) + '_' + 'interaction' in col:
                    col_list.append(col)
            tmp_interaction = pos[col_list]

            tmp_interaction['total'] = tmp_interaction.sum(axis=1)
            tmp_interaction['position'] = pos['position']
            interaction_label = str(int(i)) + '_interaction'
            pos[interaction_label] = tmp_interaction['total']
            l_list.append(interaction_label)
            ind_df.loc[i, 'interaction'] = sum(pos[interaction_label]) / len(pos)
            if len(pos[pos[r_edge] == 1])>0:
                ind_df.loc[i, 'interaction_at_edge'] = sum(pos.loc[pos[r_edge] == 1, interaction_label]) / len(pos)
                ind_df.loc[i, 'interaction_at_edge_proportion'] = sum(pos.loc[pos[r_edge] == 1, interaction_label]) / len(
                    pos[pos[r_edge] == 1])
            else:
                ind_df.loc[i, 'interaction_at_edge'] =np.nan
                ind_df.loc[i, 'interaction_at_edge_proportion']=np.nan
            if len(pos[pos[r_edge] == 0])>0:
                ind_df.loc[i, 'interaction_at_center'] = sum(pos.loc[pos[r_edge] == 0, interaction_label]) / len(pos)
                ind_df.loc[i, 'interaction_at_center_proportion'] = sum(pos.loc[pos[r_edge] == 0, interaction_label]) / len(
                    pos[pos[r_edge] == 0])
            else:
                ind_df.loc[i, 'interaction_at_center']=np.nan
                ind_df.loc[i, 'interaction_at_center_proportion']=np.nan
            if len(pos[pos[move_label_bin] == 1])>0:
                ind_df.loc[i, 'interaction_at_move'] = sum(pos.loc[pos[move_label_bin] == 1, interaction_label]) / len(pos)
                ind_df.loc[i, 'interaction_at_move_proportion'] = sum(
                    pos.loc[pos[move_label_bin] == 1, interaction_label]) / len(pos[pos[move_label_bin] == 1])
            else:
                ind_df.loc[i, 'interaction_at_move']=np.nan
                ind_df.loc[i, 'interaction_at_move_proportion']=np.nan
            if len(pos[pos[move_label_bin] == 0])>0:
                ind_df.loc[i, 'interaction_at_stop'] = sum(pos.loc[pos[move_label_bin] == 0, interaction_label]) / len(pos)
                ind_df.loc[i, 'interaction_at_stop_proportion'] = sum(
                    pos.loc[pos[move_label_bin] == 0, interaction_label]) / len(pos[pos[move_label_bin] == 0])
            else:
                ind_df.loc[i, 'interaction_at_stop']=np.nan
                ind_df.loc[i, 'interaction_at_stop_proportion']=np.nan
            if len(pos[pos[long_no_move_label_bin] == 1])>0:
                ind_df.loc[i, 'interaction_at_long_stop'] = sum(
                    pos.loc[pos[long_no_move_label_bin] == 1, interaction_label]) / len(pos)
                ind_df.loc[i, 'interaction_at_long_stop_proportion'] = sum(
                    pos.loc[pos[long_no_move_label_bin] == 1, interaction_label]) / len(pos[pos[long_no_move_label_bin] == 1])
            else:
                ind_df.loc[i, 'interaction_at_long_stop']=np.nan
                ind_df.loc[i, 'interaction_at_long_stop_proportion']=np.nan

            pos['interaction_label'] = pos[interaction_label]
            pos['value_grp'] = (pos.interaction_label.diff(1) != 0).astype('int').cumsum()

            grouped_interaction_df = pd.DataFrame({'Begin': pos.groupby('value_grp').position.first(),
                                                   'End': pos.groupby('value_grp').position.last(),
                                                   'time_duration': pos.groupby('value_grp').size(),
                                                   'interaction_counts': pos.groupby('value_grp').interaction_label.sum(),
                                                   'interactions': pos.groupby(
                                                       'value_grp').interaction_label.first()}).reset_index(drop=True)
            if len(grouped_interaction_df)>0:
                ind_df.loc[i, 'interaction_counts'] = sum(grouped_interaction_df['interactions'] > 0)
                if sum(~np.isnan(grouped_interaction_df.loc[grouped_interaction_df['interactions'] > 0, 'time_duration']))>0:
                    ind_df.loc[i, 'interaction_duration'] = np.nanmean(
                        grouped_interaction_df.loc[grouped_interaction_df['interactions'] > 0, 'time_duration']) / fps
                    ind_df.loc[i, 'interaction_members'] = np.nanmean(
                        grouped_interaction_df.loc[grouped_interaction_df['interactions'] > 0, 'interactions'])
                else:
                    ind_df.loc[i, 'interaction_duration']=np.nan
                    ind_df.loc[i, 'interaction_members']=np.nan
            else:
                ind_df.loc[i, 'interaction_counts']=np.nan
                ind_df.loc[i, 'interaction_duration']=np.nan
                ind_df.loc[i, 'interaction_members']=np.nan
            print(v_name, i, 'th dro. interaction frequency analysis finished')


            for ic in col_list:
                si = sum(tmp_interaction[ic]) / len(tmp_interaction)
                interaction_df.loc[count, 'videoname'] = v_name
                interaction_df.loc[count, 'id1'] = i
                interaction_df.loc[count, 'xlabel1'] = xlabel
                interaction_df.loc[count, 'ylabel1'] = ylabel
                ic_list = ic.split('_')
                interaction_df.loc[count, 'id2'] = int(ic_list[0])
                interaction_df.loc[count, 'xlabel2'] = 'x' + ic_list[0]
                interaction_df.loc[count, 'ylabel2'] = 'y' + ic_list[0]
                interaction_df.loc[count, 'interaction'] = si
                count = count + 1
            tmp_interaction_df = interaction_df[interaction_df['id1'] == i]
            sum_interaction = sum(tmp_interaction_df['interaction'])
            max_interaction = max(tmp_interaction_df['interaction'])
            if sum_interaction>0:
                ind_df.loc[i, 'acquaintances'] = max_interaction / sum_interaction * (n_inds - 1)
            else:
                ind_df.loc[i, 'acquaintances']=np.nan
            print(v_name, i, 'th dro. acquaintances analysis finished')
        ind_df = ind_df[['videoname', 'id', 'xlabel', 'ylabel',
                         'acquaintances',
                         'distance_space',
                         'distance_space_at_edge',
                         'distance_space_at_center',
                         'distance_space_at_move',
                         'distance_space_at_stop',
                         'SSI',
                         'SSI_at_edge',
                         'SSI_at_center',
                         'SSI_at_move',
                         'SSI_at_stop',
                         'interaction',
                         'interaction_at_edge',
                         'interaction_at_center',
                         'interaction_at_edge_proportion',
                         'interaction_at_center_proportion',
                         'interaction_at_move',
                         'interaction_at_stop',
                         'interaction_at_move_proportion',
                         'interaction_at_stop_proportion',
                         'interaction_at_long_stop',
                         'interaction_at_long_stop_proportion',
                         'interaction_counts',
                         'interaction_duration',
                         'interaction_members',
                         'betweenness_centrality',
                         'degree',
                         'clustering_coefficient',
                         'closeness_centrality',
                         'eccentricity',
                         'dominating']]
        ind_df.to_csv(fileplace_analysis + predix + '_ind_interaction.csv')
        interaction_df.to_csv(fileplace_analysis + predix + '_group_interaction_subject_to_subject.csv')
        if sum(~np.isnan(list(ind_df['acquaintances'])))>0:
            g_df.loc[0, 'acquaintances'] = np.nanmean(list(ind_df['acquaintances']))
        else:
            g_df.loc[0, 'acquaintances'] =np.nan
        if sum(~np.isnan(list(ind_df['distance_space'])))>0:
            g_df.loc[0, 'distance_space'] = np.nanmean(list(ind_df['distance_space']))
        else:
            g_df.loc[0, 'distance_space'] =np.nan
        if sum(~np.isnan(list(ind_df['distance_space_at_edge'])))>0:
            g_df.loc[0, 'distance_space_at_edge'] = np.nanmean(list(ind_df['distance_space_at_edge']))
        else:
            g_df.loc[0, 'distance_space_at_edge']=np.nan
        if sum(~np.isnan(list(ind_df['distance_space_at_center'])))>0:
            g_df.loc[0, 'distance_space_at_center'] = np.nanmean(list(ind_df['distance_space_at_center']))
        else:
            g_df.loc[0, 'distance_space_at_center'] =np.nan
        if sum(~np.isnan(list(ind_df['distance_space_at_move'])))>0:
            g_df.loc[0, 'distance_space_at_move'] = np.nanmean(list(ind_df['distance_space_at_move']))
        else:
            g_df.loc[0, 'distance_space_at_move'] =np.nan
        if sum(~np.isnan(list(ind_df['distance_space_at_stop'])))>0:
            g_df.loc[0, 'distance_space_at_stop'] = np.nanmean(list(ind_df['distance_space_at_stop']))
        else:
            g_df.loc[0, 'distance_space_at_stop'] =np.nan
        if sum(~np.isnan(list(ind_df['SSI'].astype(float))))>0:
            # print(ind_df['SSI'])
            g_df.loc[0, 'SSI'] = np.nanmean(ind_df['SSI'].astype(float))
        else:
            g_df.loc[0, 'SSI'] = np.nan
        if sum(~np.isnan(list(ind_df['SSI_at_edge'].astype(float))))>0:
            g_df.loc[0, 'SSI_at_edge'] = np.nanmean(ind_df['SSI_at_edge'].astype(float))
        else:
            g_df.loc[0, 'SSI_at_edge'] = np.nan
        if sum(~np.isnan(list(ind_df['SSI_at_center'].astype(float))))>0:
            g_df.loc[0, 'SSI_at_center'] = np.nanmean(ind_df['SSI_at_center'].astype(float))
        else:
            g_df.loc[0, 'SSI_at_center'] = np.nan
        if sum(~np.isnan(list(ind_df['SSI_at_move'].astype(float))))>0:
            g_df.loc[0, 'SSI_at_move'] = np.nanmean(ind_df['SSI_at_move'].astype(float))
        else:
            g_df.loc[0, 'SSI_at_move'] = np.nan
        if sum(~np.isnan(list(ind_df['SSI_at_stop'].astype(float))))>0:
            g_df.loc[0, 'SSI_at_stop'] = np.nanmean(ind_df['SSI_at_stop'].astype(float))
        else:
            g_df.loc[0, 'SSI_at_stop'] = np.nan
        if sum(~np.isnan(list(ind_df['interaction'].astype(float))))>0:
            g_df.loc[0, 'interaction'] = np.nanmean(ind_df['interaction'].astype(float))
        else:
            g_df.loc[0, 'interaction'] = np.nan
        if sum(~np.isnan(list(ind_df['interaction_at_edge'].astype(float))))>0:
            g_df.loc[0, 'interaction_at_edge'] = np.nanmean(ind_df['interaction_at_edge'].astype(float))
        else:
            g_df.loc[0, 'interaction_at_edge'] = np.nan
        if sum(~np.isnan(list(ind_df['interaction_at_center'].astype(float))))>0:
            g_df.loc[0, 'interaction_at_center'] = np.nanmean(ind_df['interaction_at_center'].astype(float))
        else:
            g_df.loc[0, 'interaction_at_center'] = np.nan
        if sum(~np.isnan(list(ind_df['interaction_at_edge_proportion'].astype(float))))>0:
            g_df.loc[0, 'interaction_at_edge_proportion'] = np.nanmean(ind_df['interaction_at_edge_proportion'].astype(float))
        else:
            g_df.loc[0, 'interaction_at_edge_proportion'] = np.nan
        if sum(~np.isnan(list(ind_df['interaction_at_center_proportion'].astype(float))))>0:
            g_df.loc[0, 'interaction_at_center_proportion'] = np.nanmean(ind_df['interaction_at_center_proportion'].astype(float))
        else:
            g_df.loc[0, 'interaction_at_center_proportion'] = np.nan
        if sum(~np.isnan(list(ind_df['interaction_at_move'].astype(float))))>0:
            g_df.loc[0, 'interaction_at_move'] = np.nanmean(ind_df['interaction_at_move'].astype(float))
        else:
            g_df.loc[0, 'interaction_at_move'] = np.nan
        if sum(~np.isnan(list(ind_df['interaction_at_stop'].astype(float))))>0:
            g_df.loc[0, 'interaction_at_stop'] = np.nanmean(ind_df['interaction_at_stop'].astype(float))
        else:
            g_df.loc[0, 'interaction_at_stop'] = np.nan
        if sum(~np.isnan(list(ind_df['interaction_at_move_proportion'].astype(float))))>0:
            g_df.loc[0, 'interaction_at_move_proportion'] = np.nanmean(ind_df['interaction_at_move_proportion'].astype(float))
        else:
            g_df.loc[0, 'interaction_at_move_proportion'] = np.nan
        if sum(~np.isnan(list(ind_df['interaction_at_stop_proportion'].astype(float))))>0:
            g_df.loc[0, 'interaction_at_stop_proportion'] = np.nanmean(ind_df['interaction_at_stop_proportion'].astype(float))
        else:
            g_df.loc[0, 'interaction_at_stop_proportion'] = np.nan
        if sum(~np.isnan(list(ind_df['interaction_at_long_stop'].astype(float))))>0:
            g_df.loc[0, 'interaction_at_long_stop'] = np.nanmean(ind_df['interaction_at_long_stop'].astype(float))
        else:
            g_df.loc[0, 'interaction_at_long_stop'] = np.nan
        if sum(~np.isnan(list(ind_df['interaction_counts'].astype(float))))>0:
            g_df.loc[0, 'interaction_counts'] = np.nanmean(ind_df['interaction_counts'].astype(float))
        else:
            g_df.loc[0, 'interaction_counts'] = np.nan
        if sum(~np.isnan(list(ind_df['interaction_duration'].astype(float))))>0:
            g_df.loc[0, 'interaction_duration'] = np.nanmean(ind_df['interaction_duration'].astype(float))
        else:
            g_df.loc[0, 'interaction_duration'] = np.nan
        if sum(~np.isnan(list(ind_df['interaction_members'].astype(float))))>0:
            g_df.loc[0, 'interaction_members'] = np.nanmean(ind_df['interaction_members'].astype(float))
        else:
            g_df.loc[0, 'interaction_members'] = np.nan
        if sum(~np.isnan(list(ind_df['interaction_at_long_stop_proportion'].astype(float))))>0:
            g_df.loc[0, 'interaction_at_long_stop_proportion'] = np.nanmean(ind_df['interaction_at_long_stop_proportion'].astype(float))
        else:
            g_df.loc[0, 'interaction_at_long_stop_proportion'] = np.nan
        g_df = g_df[[
            'videoname',
            'acquaintances',
            'distance_space',
            'distance_space_at_edge',
            'distance_space_at_center',
            'distance_space_at_move',
            'distance_space_at_stop',
            'SSI',
            'SSI_at_edge',
            'SSI_at_center',
            'SSI_at_move',
            'SSI_at_stop',
            'interaction',
            'interaction_at_edge',
            'interaction_at_center',
            'interaction_at_edge_proportion',
            'interaction_at_center_proportion',
            'interaction_at_move',
            'interaction_at_stop',
            'interaction_at_move_proportion',
            'interaction_at_stop_proportion',
            'interaction_at_long_stop',
            'interaction_at_long_stop_proportion',
            'interaction_counts',
            'interaction_duration',
            'interaction_members',
            'degree_assortativity_coefficient',
            'clustering_coefficient',
            'betweenness_centrality',
            'diameter',
            'degree',
            'unconnected_social_network_proportion','global_efficiency']]
        g_df.to_csv(fileplace_analysis + predix + '_avg_interaction.csv')
        moveinfo2 = pos[l_list]
        moveinfo2.to_csv(fileplace_analysis + predix + '_ind_moveinfo2.csv', index=False)
        #

        tmp_interaction_df = interaction_df[interaction_df['id1'] > interaction_df['id2']]
        G = nx.Graph()
        for i in tmp_interaction_df.index:
            G.add_weighted_edges_from([(tmp_interaction_df.loc[i, 'xlabel1'].replace('x', 'Dro.'),
                                        tmp_interaction_df.loc[i, 'xlabel2'].replace('x', 'Dro.'),
                                        tmp_interaction_df.loc[i, 'interaction'])])
        pos_pm = nx.circular_layout(G)

        Oranges_big = cm.get_cmap('Oranges', 512)
        newOranges = ListedColormap(Oranges_big(np.linspace(0.3, 0.85, 256)))
        Greys_big = cm.get_cmap('Greys', 512)
        newGreys = ListedColormap(Greys_big(np.linspace(0.3, 0.85, 256)))
        edgecolormap = newOranges
        nodecolormap = Greys_big
        nodes, nodes_degree = zip(*G.degree)
        nodes = list(nodes)
        edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
        weights = np.round(np.array(weights), 3)
        edgelabels = dict(zip(edges, weights))
        nlabels = dict(zip(nodes, nodes))
        plt.figure(figsize=(6, 6))
        nx.draw(G, pos_pm, edge_color=weights, edge_cmap=edgecolormap, nodelist=nodes, cmap=nodecolormap)
        nx.draw_networkx_labels(G, pos_pm, nlabels, font_size=12)
        nx.draw_networkx_edge_labels(G, pos_pm, edge_labels=edgelabels, alpha=0.9, rotate=True, label_pos=0.3)
        # plt.colorbar(edgemap, shrink=0.3, label='Interactions')
        plt.axis('off')
        # plt.show()
        # plt.savefig(fileplace_analysis + '/PM_network_withintaxa.pdf',dpi=600, bbox_inches='tight')
        plt.savefig(fileplace_analysis + predix + '_group_interaction.jpg')

        dis_label_list = []
        for i in range(n_inds):
            dis_ij = str(int(i)) + '_dis_space'
            dis_label_list.append(dis_ij)
        pos['avg_dis'] = pos[dis_label_list].mean(axis=1)
        frame_list = pos['position'] / fps / 60  # in minute
        distance_space = pos['avg_dis']
        plt.figure(figsize=(18, 6))
        plt.plot(frame_list, distance_space, alpha=0.5)
        plt.xlabel('Time(min)', fontsize=10)
        plt.ylabel('Distance(mm)', fontsize=10)
        plt.title(v_name + ' Avarage Distance from other drosophila')
        plt.tight_layout()
        plt.savefig(fileplace_analysis + predix + '_social_space.png')
        plt.close()



def ind_motion_analysis(predix,tracefile,n_inds,x,y,r,saveplace,videoname,real_r,x_max,y_max,fps,r_thresh,max_v_thresh,area_thresh,sensing_area,move_thresh,area_search_time_thresh,long_stop_thresh,angular_velocity_window,track_straightness_window,scaling):
    fileplace_analysis=saveplace+'/'
    x_min=0
    y_min=0
    r_thresh=r_thresh/real_r
    csv_name=tracefile
    v_name=videoname
    cycle_area=np.pi*r*r
    scaling_to_mm=real_r/r

    l_list=[]
    xlabel_list=[]
    ylabel_list=[]
    pos=pd.read_csv(csv_name,header=0)

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
                                 'r_dist','r_edge',
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
                               'r_dist','r_edge',
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
        plt.savefig(fileplace_analysis + predix+'_'+xlabel+'_'+ylabel + '_motion_trace.png')
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
        plt.savefig(fileplace_analysis + predix+'_'+xlabel+'_'+ylabel + '_motion_velocity.png')
        plt.close()


    ind_df_tocsv=ind_df[['videoname','id','xlabel','ylabel',
                         'search_area_time','search_area','search_area_unit_move',
                         'velocity_threshed','velocity_at_edge','velocity_at_center',
                         'max_velocity_threshed','max_velocity_at_edge','max_velocity_at_center',
                         'total_move_length_threshed','movelength_at_edge','movelength_at_center',
                         'move_time_threshed','move_proportion_at_edge','move_proportion_at_center',
                         'r_dist','r_edge',
                         'tracks_num','tracks_duration','tracks_length',
                         'long_stop_num','stop_duration','long_stop_duration',
                         'track_straightness','track_straightness_non_edge','track_straightness_at_edge',
                         'angular_velocity','angular_velocity_non_edge','angular_velocity_at_edge',
                         'max_angular_velocity','max_angular_velocity_non_edge','max_angular_velocity_at_edge',
                         'meander','meander_non_edge','meander_at_edge',
                         'max_meander','max_meander_non_edge','max_meander_at_edge'
                         ]]
    ind_df_tocsv.to_csv(fileplace_analysis + predix + '_ind_motion.csv')
    g_df.loc[0,'videoname']=v_name

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
                     'r_dist','r_edge',
                     'tracks_num','tracks_duration','tracks_length',
                     'long_stop_num','stop_duration','long_stop_duration',
                     'track_straightness','track_straightness_non_edge','track_straightness_at_edge',
                     'angular_velocity','angular_velocity_non_edge','angular_velocity_at_edge',
                     'max_angular_velocity','max_angular_velocity_non_edge','max_angular_velocity_at_edge',
                     'meander','meander_non_edge','meander_at_edge',
                     'max_meander','max_meander_non_edge','max_meander_at_edge'
                     ]]

    g_df_tocsv.to_csv(fileplace_analysis + predix + '_avg_motion.csv')

    moveinfo=pos[l_list+['position']]
    moveinfo.to_csv(fileplace_analysis + predix + '_ind_moveinfo.csv',index=False)

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
    plt.savefig(fileplace_analysis + predix+ '_motion_trace.png')
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
    plt.savefig(fileplace_analysis + predix + '_motion_velocity.png')
    plt.close()




def find_center(A,B,C):
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
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

def generate_bodysize_image(videoname,saveplace,predix,imgs_for_body_size_measure):
    # background_generate_frames=100
    N_file=saveplace+'/img_for_body_size_measure/'
    if not os.path.exists(N_file):
        os.makedirs(N_file)
    cap = cv2.VideoCapture(videoname)
    frames_num=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(imgs_for_body_size_measure):
        f=random.randint(1,int(frames_num))
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(f))
        rval, frame = cap.read()
        cv2.imwrite(N_file+predix+'_'+str(f).zfill(7)+'.jpg',frame)
    cap.release()
    return None
def getbackground(videoname,fileplace,predix,background_generate_frames):
    # background_generate_frames=100
    startframe=5
    input_vidpath = fileplace+'/' + videoname
    print(input_vidpath)
    cap = cv2.VideoCapture(input_vidpath)
    total_frame=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    endframe=total_frame-5
    sample_interval=int(min(total_frame,endframe-startframe)/background_generate_frames)
    img_list = []
    flist=np.array(range(startframe,endframe,sample_interval))
    for f in flist:
        # print(f)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_list.append(img)
    img_list=np.asarray(img_list)
    b_list=np.median(img_list,axis=0)
    cv2.imwrite(fileplace+'/'+predix+'_background.jpg', b_list)
    cap.release()
    return None

def Contrast_and_Brightness(alpha, beta, img):
    blank = np.zeros(img.shape, img.dtype)
    # dst = alpha * img + beta * blank
    dst = cv2.addWeighted(img, alpha, blank, 1-alpha, beta)
    return dst


def removebackground(x, y, r, videoname, predix, saveplace, imgname):
    remove=False
    try:
        codec = 'mp4v'
        input_vidpath =  videoname
        modified_vidpath =saveplace+'/'+predix+'_background_removed.mp4'
        cap = cv2.VideoCapture(input_vidpath)
        fps=cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        x_dim=int(cap.read()[1].shape[0])
        y_dim=int(cap.read()[1].shape[1])
        output_framesize = (y_dim,x_dim)
        cap.release()
        cap = cv2.VideoCapture(input_vidpath)
        out = cv2.VideoWriter(filename=modified_vidpath, fourcc=fourcc, fps=fps, frameSize=output_framesize, isColor=False)
        b_list=cv2.imread(imgname)
        b_list=cv2.cvtColor(b_list, cv2.COLOR_BGR2GRAY)
        bright=np.mean(b_list)
        last=0
        mask = np.zeros(shape=[int(x_dim),int(y_dim)])
        cv2.circle(mask, (int(x),int(y)), int(r), 255, -1)

        while (True):
            ret, frame = cap.read()
            this = cap.get(1)
            if ret == True:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                img_bright=np.mean(img)
                img = np.asarray(img)
                img=np.clip((img+bright-img_bright),0,255)
                img=img.astype('uint8')
                img_diff = cv2.absdiff(b_list, img)
                img_diff[mask < 250] = 0

                # b=np.percentile(np.asarray(img_diff), 99.98)
                # alpha=200/b
                # beta=0
                # img_diff=Contrast_and_Brightness(alpha, beta, img_diff)

                out.write(img_diff)
            if last >= this:
                break
            last = this
            if this%90==0:
                print(this/30)
        cap.release()
        out.release()
        remove=True
    except:
        pass
    return remove

def clipvideo(videoname, outname,start,end):
    remove=False
    try:
        codec = 'mp4v'
        input_vidpath =  videoname
        modified_vidpath = outname
        cap = cv2.VideoCapture(input_vidpath)
        fps=cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        x_dim=int(cap.read()[1].shape[0])
        y_dim=int(cap.read()[1].shape[1])
        output_framesize = (y_dim,x_dim)
        cap.release()
        cap = cv2.VideoCapture(input_vidpath)
        out = cv2.VideoWriter(filename=modified_vidpath, fourcc=fourcc, fps=fps, frameSize=output_framesize, isColor=True)
        last=0
        start=int(start*fps)
        end=int(end*fps)
        while (True):
            ret, frame = cap.read()
            this = cap.get(1)
            if ret == True:
                if this>=start:
                    # img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    out.write(frame)
                    if this%90==0:
                        print(this/30)
            if this>=end:
                break
            if last >= this:
                break
            last = this

        cap.release()
        out.release()
        remove=True
    except:
        pass
    return remove


class query_window(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = Ui_Videotrack()
        self.ui.setupUi(self)
        self.ui.label_videoname.setWordWrap(True)
        self.ui.label_background_img.setWordWrap(True)
        self.ui.label_clipedvideo.setWordWrap(True)
        self.ui.label_videoname.setWordWrap(True)

        self.ui.lineEdit_starttime.setValidator(QIntValidator())
        self.ui.lineEdit_endtime.setValidator(QIntValidator())
        self.ui.lineEdit_num_s3.setValidator(QIntValidator())
        self.ui.lineEdit_xdim_s3.setValidator(QIntValidator())
        self.ui.lineEdit_ydim_s3.setValidator(QIntValidator())
        self.ui.lineEdit_r_s3.setValidator(QIntValidator())

        self.ui.lineEdit_projectname_3.setValidator(QIntValidator())
        self.ui.lineEdit_projectname_4.setValidator(QIntValidator())
        self.ui.lineEdit_projectname_5.setValidator(QIntValidator())
        self.ui.lineEdit_projectname_6.setValidator(QIntValidator())
        self.ui.lineEdit_projectname_7.setValidator(QIntValidator())
        self.ui.lineEdit_projectname_8.setValidator(QDoubleValidator())
        self.ui.lineEdit_projectname_9.setValidator(QDoubleValidator())

        self.ui.lineEdit_projectname_10.setValidator(QIntValidator())
        self.ui.lineEdit_projectname_11.setValidator(QDoubleValidator())
        self.ui.lineEdit_projectname_12.setValidator(QDoubleValidator())
        self.ui.lineEdit_projectname_13.setValidator(QDoubleValidator())
        self.ui.lineEdit_projectname_14.setValidator(QDoubleValidator())
        self.ui.lineEdit_projectname_15.setValidator(QDoubleValidator())
        self.ui.lineEdit_projectname_16.setValidator(QDoubleValidator())
        self.ui.lineEdit_projectname_17.setValidator(QDoubleValidator())
        self.ui.lineEdit_projectname_18.setValidator(QDoubleValidator())
        self.ui.lineEdit_projectname_19.setValidator(QDoubleValidator())

        # step 1
        self.ui.pushButton_openvideo.clicked.connect(self.open_video_s1)
        self.ui.pushButton_select_save.clicked.connect(self.select_save_s1)
        self.ui.checkBox_30.stateChanged.connect(self.check_30)
        self.ui.pushButton_run.clicked.connect(self.run_s1)
        self.ui.lineEdit_projectname_s1.changeEvent=self.projectname_s1_changed
        # step 2
        self.ui.pushButton_openimg_s2.clicked.connect(self.openimg_s2)
        self.ui.background_label_s2.setCursor(Qt.CrossCursor)
        self.ui.background_label_s2.mousePressEvent = self.getPos
        self.ui.pushButton_reset_s2.clicked.connect(self.reset_points_s2)
        self.ui.pushButton_find_cyecle_s2.clicked.connect(self.get_center_s2)
        self.ui.pushButton_open_cliped_video_s2.clicked.connect(self.open_video_s2)
        self.ui.pushButton_selectsave_s2.clicked.connect(self.open_saveplace_s2)
        self.ui.pushButton_run_s2.clicked.connect(self.run_s2)
        # step 3
        self.ui.pushButton_opentrack_s3.clicked.connect(self.opentrack_s3)
        self.ui.pushButton_open_video_s3.clicked.connect(self.openvideo_s3)
        self.ui.pushButton_check_s3.clicked.connect(self.check_s3)
        self.ui.pushButton_select_save_s3.clicked.connect(self.open_saveplace_s3)
        self.ui.pushButton_run_s3.clicked.connect(self.run_s3)

    def run_s3(self):
        predix=self.ui.lineEdit_projectname.text()
        if predix=='':
            QMessageBox.information(self, 'Message', 'Please name the project, exit', QMessageBox.Ok)
            return None
        videoname=self.ui.lineEdit_openvideo.text()
        if not os.path.exists(videoname):
            QMessageBox.information(self, 'Message', 'Videofile not valid, exit', QMessageBox.Ok)
            return None
        trackfilename=self.ui.lineEdit_trackfile_s3.text()
        if not os.path.exists(trackfilename):
            QMessageBox.information(self, 'Message', 'Trackfile not valid, exit', QMessageBox.Ok)
            return None
        num=self.ui.lineEdit_num_s3.text()
        if num=='':
            QMessageBox.information(self, 'Message', 'Please input the number of drosophilas, exit', QMessageBox.Ok)
            return None
        num=int(num)
        if num<1:
            QMessageBox.information(self, 'Message', 'The number of drosophilas should be large than 0, exit', QMessageBox.Ok)
            return None
        xdim=self.ui.lineEdit_xdim_s3.text()
        if xdim=='':
            QMessageBox.information(self, 'Message', 'Please input the xdim in center of cycle, exit', QMessageBox.Ok)
            return None
        xdim=int(xdim)
        if xdim<1:
            QMessageBox.information(self, 'Message', 'The xdim in center of cycle should be large than 0, exit', QMessageBox.Ok)
            return None
        ydim=self.ui.lineEdit_ydim_s3.text()
        if ydim=='':
            QMessageBox.information(self, 'Message', 'Please input the ydim in center of cycle, exit', QMessageBox.Ok)
            return None
        ydim=int(ydim)
        if ydim<1:
            QMessageBox.information(self, 'Message', 'The ydim in center of cycle should be large than 0, exit', QMessageBox.Ok)
            return None
        r=self.ui.lineEdit_r_s3.text()
        if r=='':
            QMessageBox.information(self, 'Message', 'Please input the radius in center of cycle, exit', QMessageBox.Ok)
            return None
        r=int(r)
        if r<1:
            QMessageBox.information(self, 'Message', 'The radius in center of cycle should be large than 0, exit', QMessageBox.Ok)
            return None
        saveplace=self.ui.lineEdit_select_save_s3.text()
        if not os.path.exists(saveplace):
            QMessageBox.information(self, 'Message', 'Saveplace not valid, exit', QMessageBox.Ok)
            return None
        real_r=float(self.ui.lineEdit_projectname_9.text())/2
        x_max=int(float(self.ui.lineEdit_projectname_6.text()))
        y_max=int(float(self.ui.lineEdit_projectname_7.text()))
        fps=int(float(self.ui.lineEdit_projectname_3.text()))
        r_thresh=float(self.ui.lineEdit_projectname_12.text())
        max_v_thresh=float(self.ui.lineEdit_projectname_13.text())
        area_thresh=float(self.ui.lineEdit_projectname_18.text())
        sensing_area=float(self.ui.lineEdit_projectname_10.text())
        move_thresh=float(self.ui.lineEdit_projectname_11.text())
        area_search_time_thresh=float(self.ui.lineEdit_projectname_17.text())
        long_stop_thresh=float(self.ui.lineEdit_projectname_16.text())
        angular_velocity_window=float(self.ui.lineEdit_projectname_14.text())
        track_straightness_window=float(self.ui.lineEdit_projectname_15.text())
        scaling=float(self.ui.lineEdit_projectname_8.text())
        scale_to_interaction=float(self.ui.lineEdit_projectname_19.text())
        ssi_bin=float(self.ui.lineEdit_projectname_20.text())
        network_size=float(self.ui.lineEdit_projectname_21.text())
        ind_motion_analysis(predix,trackfilename,num,xdim,ydim,r,saveplace,videoname,real_r,x_max,y_max,fps,r_thresh,max_v_thresh,area_thresh,sensing_area,move_thresh,area_search_time_thresh,long_stop_thresh,angular_velocity_window,track_straightness_window,scaling)
        if num>1:
            group_interaction_analysis(saveplace,trackfilename,videoname,xdim,ydim,r,num,predix,real_r,sensing_area,scale_to_interaction,ssi_bin,fps,network_size,scaling)
        if self.ui.checkBox_generated_notedvideo.checkState()==2:
            annoateVideo(saveplace,trackfilename,videoname,num,predix,scaling)
        QMessageBox.information(self, 'Message', 'Analysis done\n Results stored at '+saveplace, QMessageBox.Ok)




    def open_saveplace_s3(self):
        directory1 = QFileDialog.getExistingDirectory(self, "select folder", "/")
        print(directory1)
        if os_p == 'Windows':
            tmpdatafilename = directory1.replace('/', '\\')
        else:
            tmpdatafilename = directory1
        self.ui.lineEdit_select_save_s3.setText(tmpdatafilename)


    def check_s3(self):
        videoname=self.ui.lineEdit_openvideo.text()
        if not os.path.exists(videoname):
            QMessageBox.information(self, 'Message', 'Videofile not valid, exit', QMessageBox.Ok)
            return None
        trackfilename=self.ui.lineEdit_trackfile_s3.text()
        if not os.path.exists(trackfilename):
            QMessageBox.information(self, 'Message', 'Trackfile not valid, exit', QMessageBox.Ok)
            return None
        num=self.ui.lineEdit_num_s3.text()
        if num=='':
            QMessageBox.information(self, 'Message', 'Please input the number of drosophilas, exit', QMessageBox.Ok)
            return None
        num=int(num)
        if num<1:
            QMessageBox.information(self, 'Message', 'The number of drosophilas should be large than 0, exit', QMessageBox.Ok)
            return None
        xdim=self.ui.lineEdit_xdim_s3.text()
        if xdim=='':
            QMessageBox.information(self, 'Message', 'Please input the xdim in center of cycle, exit', QMessageBox.Ok)
            return None
        xdim=int(xdim)
        if xdim<1:
            QMessageBox.information(self, 'Message', 'The xdim in center of cycle should be large than 0, exit', QMessageBox.Ok)
            return None
        ydim=self.ui.lineEdit_ydim_s3.text()
        if ydim=='':
            QMessageBox.information(self, 'Message', 'Please input the ydim in center of cycle, exit', QMessageBox.Ok)
            return None
        ydim=int(ydim)
        if ydim<1:
            QMessageBox.information(self, 'Message', 'The ydim in center of cycle should be large than 0, exit', QMessageBox.Ok)
            return None
        r=self.ui.lineEdit_r_s3.text()
        if r=='':
            QMessageBox.information(self, 'Message', 'Please input the radius in center of cycle, exit', QMessageBox.Ok)
            return None
        r=int(r)
        if r<1:
            QMessageBox.information(self, 'Message', 'The radius in center of cycle should be large than 0, exit', QMessageBox.Ok)
            return None
        red = (48, 48, 255)
        green = (34, 139, 34)
        yellow = (0, 255, 255)
        color=[red,green,yellow]
        font = cv2.FONT_HERSHEY_SIMPLEX
        trace=pd.read_csv(trackfilename)
        cap = cv2.VideoCapture(videoname)
        total_frame=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame=int(total_frame/2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(current_frame))
        subject_list= ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H','I','J','K','L','M','N']
        if cap.isOpened():
            rval, frame = cap.read()

            cv2.circle(frame, (int(xdim),int(ydim)), int(r), (255, 255, 255), 2)
            cv2.circle(frame, (int(xdim),int(ydim)), 3, red, -1, cv2.LINE_AA)

            for i in range(num):
                xlabel='x'+str(i)
                ylabel='y'+str(i)
                dx=int(trace.loc[current_frame,xlabel])
                dy=int(trace.loc[current_frame,ylabel])
                cv2.circle(frame, (dx,dy), 3, yellow, -1, cv2.LINE_AA)
                cv2.putText(frame, subject_list[i], (dx+5,dy+15), font, 1,(255,255,255), 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (400, 300))
            img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            self.ui.label_videopreview_s3.setPixmap(QPixmap.fromImage(img))
        else:
            QMessageBox.information(self, 'Message', 'Video file not valid', QMessageBox.Ok)
        cap.release()



    def openvideo_s3(self):
        dig = QFileDialog()
        dig.setNameFilters(["Clipped video file(*.mp4)"])
        dig.setFileMode(QFileDialog.ExistingFile)
        dig.setFilter(QDir.Files)
        if dig.exec_():
            filenames = dig.selectedFiles()
            if os_p == 'Windows':
                tmpdatafilename = filenames[0].replace('/', '\\')
            else:
                tmpdatafilename = filenames[0]

            cap = cv2.VideoCapture(tmpdatafilename)
            total_frame=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            current_frame=int(total_frame/2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(current_frame))
            if cap.isOpened():
                self.ui.lineEdit_openvideo.setText(tmpdatafilename)
                rval, frame = cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = cv2.resize(frame, (400, 300))
                img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
                self.ui.label_videopreview_s3.setPixmap(QPixmap.fromImage(img))
            else:
                QMessageBox.information(self, 'Message', 'Video file not valid', QMessageBox.Ok)
            cap.release()
            self.ui.lineEdit_xdim_s3.setText('')
            self.ui.lineEdit_ydim_s3.setText('')
            self.ui.lineEdit_r_s3.setText('')

    def opentrack_s3(self):
        dig = QFileDialog()
        dig.setNameFilters(["Track file(*.csv)"])
        dig.setFileMode(QFileDialog.ExistingFile)
        dig.setFilter(QDir.Files)
        if dig.exec_():
            filenames = dig.selectedFiles()
            if os_p == 'Windows':
                tmpdatafilename = filenames[0].replace('/', '\\')
            else:
                tmpdatafilename = filenames[0]
            self.ui.lineEdit_trackfile_s3.setText(tmpdatafilename)
            self.ui.lineEdit_xdim_s3.setText('')
            self.ui.lineEdit_ydim_s3.setText('')
            self.ui.lineEdit_r_s3.setText('')


    def run_s2(self):
        predix=self.ui.lineEdit_projectname_s2.text()
        if len(predix)==0:
            QMessageBox.information(self, 'Message', 'Project name is None, exit', QMessageBox.Ok)
            return None
        videoname=self.ui.lineEdit_clipedvideo_s2.text()
        if not os.path.exists(videoname):
            QMessageBox.information(self, 'Message', 'Videofile not valid, exit', QMessageBox.Ok)
            return None
        saveplace=self.ui.lineEdit_select_save_s2.text()
        if not os.path.exists(saveplace):
            QMessageBox.information(self, 'Message', 'Saveplace not valid, exit', QMessageBox.Ok)
            return None
        imgname=self.ui.lineEdit_img_s2.text()
        if not os.path.exists(saveplace):
            QMessageBox.information(self, 'Message', 'Background image not valid, exit', QMessageBox.Ok)
            return None
        if self.ui.label_xdim_p3_s2.text()=='None':
            QMessageBox.information(self, 'Message', 'Please pick three points at the edge of the circle by mouse click, exit', QMessageBox.Ok)
            return None
        if self.ui.label_xdim_r_s2.text()=='None':
            QMessageBox.information(self, 'Message', 'Please find the cycle center, exit', QMessageBox.Ok)
            return None
        x,y,r=int(float(self.ui.label_xdim_r_s2.text())),int(float(self.ui.label_ydim_r_s2.text())),int(float(self.ui.label_r_s2.text()))
        remove=removebackground(x, y, r, videoname, predix, saveplace, imgname)
        if remove and (os.path.exists(saveplace+'/'+predix+'_background_removed.mp4')):
            QMessageBox.information(self, 'Message', 'Background removed success\n Video stored at '+saveplace+'/'+predix+'_background_removed.mp4', QMessageBox.Ok)
            cycle=pd.DataFrame(columns={'name','value'})
            cycle.loc[0,'name']='x'
            cycle.loc[0,'value']=x
            cycle.loc[1,'name']='y'
            cycle.loc[1,'value']=y
            cycle.loc[2,'name']='r'
            cycle.loc[2,'value']=r
            cycle.to_csv(saveplace+'/'+predix+'_cycle.csv')
            QMessageBox.information(self, 'Message', 'Cycle information stored at '+saveplace+'/'+predix+'_cycle.csv', QMessageBox.Ok)

            return None
        else:
            QMessageBox.information(self, 'Message', 'Video generation failed, exit', QMessageBox.Ok)
            return None


    def open_saveplace_s2(self):
        directory1 = QFileDialog.getExistingDirectory(self, "select folder", "/")
        print(directory1)
        if os_p == 'Windows':
            tmpdatafilename = directory1.replace('/', '\\')
        else:
            tmpdatafilename = directory1
        self.ui.lineEdit_select_save_s2.setText(tmpdatafilename)

    def open_video_s2(self):
        dig = QFileDialog()
        dig.setNameFilters(["Clipped video file(*.mp4)"])
        dig.setFileMode(QFileDialog.ExistingFile)
        dig.setFilter(QDir.Files)
        if dig.exec_():
            filenames = dig.selectedFiles()
            if os_p == 'Windows':
                tmpdatafilename = filenames[0].replace('/', '\\')
            else:
                tmpdatafilename = filenames[0]
            self.ui.lineEdit_clipedvideo_s2.setText(tmpdatafilename)


    def get_center_s2(self):
        xylist=[]
        if self.ui.label_xdim_s2.text()!='None':
            xylist.append([int(float(self.ui.label_xdim_s2.text())),int(float(self.ui.label_ydim_s2.text()))])
        if self.ui.label_xdim_p2_s2.text()!='None':
            xylist.append([int(float(self.ui.label_xdim_p2_s2.text())),int(float(self.ui.label_ydim_p2_s2.text()))])
        if self.ui.label_xdim_p3_s2.text()!='None':
            xylist.append([int(float(self.ui.label_xdim_p3_s2.text())),int(float(self.ui.label_ydim_p3_s2.text()))])
        if len(xylist)<3:
            QMessageBox.information(self, 'Message', 'Please select three points at the edge of the cycle, exit', QMessageBox.Ok)
            return None
        if len(xylist)==3:
            red = (48, 48, 255)
            green = (34, 139, 34)
            yellow = (0, 255, 255)
            tmpdatafilename=self.ui.lineEdit_img_s2.text()
            if os.path.exists(tmpdatafilename):
                img = cv2.imread(tmpdatafilename)
                P,R=find_center(xylist[0],xylist[1],xylist[2])
                x,y=P
                x=int(x)
                y=int(y)
                R=int(R)
                cv2.circle(img, (int(x),int(y)), int(R), (255, 255, 255), 2)
                cv2.circle(img, (int(x),int(y)), 3, red, -1, cv2.LINE_AA)
                frame = cv2.resize(img, (640, 480))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
                self.ui.background_label_s2.setPixmap(QPixmap.fromImage(img))
                self.ui.label_xdim_r_s2.setText(str(x))
                self.ui.label_ydim_r_s2.setText(str(y))
                self.ui.label_r_s2.setText(str(R))

            else:
                QMessageBox.information(self, 'Message', 'Image not found, exit', QMessageBox.Ok)
                return None

    def reset_points_s2(self):

        self.ui.label_xdim_s2.setText('None')
        self.ui.label_ydim_s2.setText('None')

        self.ui.label_xdim_p2_s2.setText('None')
        self.ui.label_ydim_p2_s2.setText('None')

        self.ui.label_xdim_p3_s2.setText('None')
        self.ui.label_ydim_p3_s2.setText('None')
        self.ui.label_xdim_r_s2.setText('None')
        self.ui.label_ydim_r_s2.setText('None')
        self.ui.label_r_s2.setText('None')

        tmpdatafilename=self.ui.lineEdit_img_s2.text()
        if os.path.exists(tmpdatafilename):
            img = cv2.imread(tmpdatafilename)
            frame = cv2.resize(img, (640, 480))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            self.ui.background_label_s2.setPixmap(QPixmap.fromImage(img))

    def getPos(self , event):
        xylist=[]
        if self.ui.label_xdim_s2.text()!='None':
            xylist.append((int(float(self.ui.label_xdim_s2.text())),int(float(self.ui.label_ydim_s2.text()))))
        if self.ui.label_xdim_p2_s2.text()!='None':
            xylist.append((int(float(self.ui.label_xdim_p2_s2.text())),int(float(self.ui.label_ydim_p2_s2.text()))))
        if self.ui.label_xdim_p3_s2.text()!='None':
            xylist.append((int(float(self.ui.label_xdim_p3_s2.text())),int(float(self.ui.label_ydim_p3_s2.text()))))

        red = (48, 48, 255)
        green = (34, 139, 34)
        yellow = (0, 255, 255)
        color=[red,green,yellow]
        font = cv2.FONT_HERSHEY_SIMPLEX
        x = int(event.pos().x()*1.6)
        y = int(event.pos().y()*1.6)
        # print(x,y)
        if len(xylist)<3:
            xylist.append((x,y))
            tmpdatafilename=self.ui.lineEdit_img_s2.text()
            frame = cv2.imread(tmpdatafilename)
            for i in range(len(xylist)):
                cv2.circle(frame, xylist[i], 3, color[i], -1, cv2.LINE_AA)
            frame = cv2.resize(frame, (640, 480))
                # cv2.putText(frame, 'point '+str(i+1), xylist[i], font, 0.5, (0, 0, 0), 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            self.ui.background_label_s2.setPixmap(QPixmap.fromImage(img))
            if len(xylist)==1:
                self.ui.label_xdim_s2.setText(str(x))
                self.ui.label_ydim_s2.setText(str(y))
            if len(xylist)==2:
                self.ui.label_xdim_p2_s2.setText(str(x))
                self.ui.label_ydim_p2_s2.setText(str(y))
            if len(xylist)==3:
                self.ui.label_xdim_p3_s2.setText(str(x))
                self.ui.label_ydim_p3_s2.setText(str(y))

    def projectname_s1_changed(self,event):
        self.ui.label_background_img.setText('None')
        self.ui.label_clipedvideo.setText('None')

    def openimg_s2(self):
        dig = QFileDialog()
        dig.setNameFilters(["background file(*.jpg)"])
        dig.setFileMode(QFileDialog.ExistingFile)
        dig.setFilter(QDir.Files)
        if dig.exec_():
            filenames = dig.selectedFiles()
            if os_p == 'Windows':
                tmpdatafilename = filenames[0].replace('/', '\\')
            else:
                tmpdatafilename = filenames[0]
            img = cv2.imread(tmpdatafilename)
            frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (640, 480))
            img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            self.ui.background_label_s2.setPixmap(QPixmap.fromImage(img))
            self.ui.lineEdit_img_s2.setText(tmpdatafilename)
            self.ui.label_xdim_s2.setText('None')
            self.ui.label_ydim_s2.setText('None')

            self.ui.label_xdim_p2_s2.setText('None')
            self.ui.label_ydim_p2_s2.setText('None')

            self.ui.label_xdim_p3_s2.setText('None')
            self.ui.label_ydim_p3_s2.setText('None')
            self.ui.label_xdim_r_s2.setText('None')
            self.ui.label_ydim_r_s2.setText('None')
            self.ui.label_r_s2.setText('None')
            self.ui.lineEdit_clipedvideo_s2.setText('')



    def open_video_s1(self):
        dig = QFileDialog()
        dig.setNameFilters(["mp4 video file(*.mp4)","mkv video file(*.mkv)"])
        dig.setFileMode(QFileDialog.ExistingFile)
        dig.setFilter(QDir.Files)
        if dig.exec_():
            filenames = dig.selectedFiles()
            if os_p == 'Windows':
                tmpdatafilename = filenames[0].replace('/', '\\')
            else:
                tmpdatafilename = filenames[0]

            cap = cv2.VideoCapture(tmpdatafilename)
            total_frame=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            current_frame=int(total_frame/2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(current_frame))
            if cap.isOpened():
                self.ui.lineEdit_videoname.setText(tmpdatafilename)
                rval, frame = cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = cv2.resize(frame, (400, 300))
                img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
                self.ui.img_label_s1.setPixmap(QPixmap.fromImage(img))
                fps=cap.get(cv2.CAP_PROP_FPS)
                ## Video writer class to output video with contour and centroid of tracked object(s)
                # make sure the frame size matches size of array 'final'
                x_dim=int(cap.read()[1].shape[1])
                y_dim=int(cap.read()[1].shape[0])
                length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                time_duration=length/fps
                self.ui.label_videoname.setText(tmpdatafilename)
                self.ui.label_total_length.setText(str(time_duration)+' s')
                self.ui.label_totalframe.setText(str(length)+' Frames')
                self.ui.label_xdim.setText(str(x_dim))
                self.ui.label_ydim.setText(str(y_dim))
            else:
                QMessageBox.information(self, 'Message', 'Video file not valid', QMessageBox.Ok)
            cap.release()
            self.ui.label_background_img.setText('None')
            self.ui.label_clipedvideo.setText('None')



    def select_save_s1(self):
        directory1 = QFileDialog.getExistingDirectory(self, "select folder", "/")
        print(directory1)
        if os_p == 'Windows':
            tmpdatafilename = directory1.replace('/', '\\')
        else:
            tmpdatafilename = directory1
        self.ui.lineEdit_saveplace.setText(tmpdatafilename)
        self.ui.label_background_img.setText('None')
        self.ui.label_clipedvideo.setText('None')


    def check_30(self):
        # print(self.ui.checkBox_30.checkState())
        if self.ui.checkBox_30.checkState()==0:
            self.ui.lineEdit_endtime.setDisabled(False)
        if self.ui.checkBox_30.checkState()==2:
            self.ui.lineEdit_endtime.setDisabled(True)
    def run_s1(self):
        predix=self.ui.lineEdit_projectname_s1.text()
        if len(predix)==0:
            QMessageBox.information(self, 'Message', 'Project name is None, exit', QMessageBox.Ok)
            return None
        videoname=self.ui.lineEdit_videoname.text()
        if not os.path.exists(videoname):
            QMessageBox.information(self, 'Message', 'Videofile not valid, exit', QMessageBox.Ok)
            return None
        saveplace=self.ui.lineEdit_saveplace.text()
        if not os.path.exists(saveplace):
            QMessageBox.information(self, 'Message', 'Saveplace not valid, exit', QMessageBox.Ok)
            return None
        try:
            starttime=int(self.ui.lineEdit_starttime.text())
        except:
            starttime=0
            QMessageBox.information(self, 'Message', 'Start time not valid, exit', QMessageBox.Ok)
            return None
        try:
            totaltime=int(float(self.ui.label_total_length.text().replace(' s','')))
        except:
            totaltime=0
            QMessageBox.information(self, 'Message', 'Video file not valid, exit', QMessageBox.Ok)
            return None
        endtime=0
        if self.ui.checkBox_30.checkState()==0:
            try:
                endtime=int(self.ui.lineEdit_endtime.text())
            except:
                endtime=0
                QMessageBox.information(self, 'Message', 'End time not valid, exit', QMessageBox.Ok)
                return None
            if endtime>totaltime:
                QMessageBox.information(self, 'Message', 'End time is larger than total video length, exit', QMessageBox.Ok)
                return None
        if self.ui.checkBox_30.checkState()==2:
            endtime=starttime+30*60
            if endtime>totaltime:
                QMessageBox.information(self, 'Message', 'Clipped video length is smaller than 30mins, exit', QMessageBox.Ok)
                return None
        # clip video

        clipvideo(videoname, saveplace+'/'+predix+'.mp4',starttime,endtime)
        # clip=VideoFileClip(videoname)
        # clip=clip.subclip(starttime,endtime)
        # videoname=predix+'.mp4'
        # clip.to_videofile(saveplace+'/'+videoname, fps=30)

        cap = cv2.VideoCapture(saveplace+'/'+predix+'.mp4')
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(5))
        if cap.isOpened():
            pass
        else:
            QMessageBox.information(self, 'Message', 'Video clip failed', QMessageBox.Ok)
            return None
        # generate background
        background_generate_frames=int(float(self.ui.lineEdit_projectname_4.text()))
        getbackground(predix+'.mp4',saveplace,predix,background_generate_frames)
        imgs_for_body_size_measure=int(float(self.ui.lineEdit_projectname_5.text()))
        generate_bodysize_image(videoname,saveplace,predix,imgs_for_body_size_measure)
        if not os.path.exists(saveplace+'/'+predix+'_background.jpg'):
            QMessageBox.information(self, 'Message', 'Background image generation failed', QMessageBox.Ok)
            return None
        self.ui.label_background_img.setText(saveplace+'/'+predix+'_background.jpg')
        self.ui.label_clipedvideo.setText(saveplace+'/'+predix+'.mp4')
        QMessageBox.information(self, 'Message', 'Finished', QMessageBox.Ok)

if __name__ == '__main__':
    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    window = query_window()
    window.show()
    sys.exit(app.exec_())
