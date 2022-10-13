import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import networkx as nx
from matplotlib import cm
from matplotlib.colors import ListedColormap
import warnings
# import platform
# import timeout_decorator
# sys = platform.system()
warnings.filterwarnings("ignore")
import sys
import math
#
fileplace = sys.argv[1]
metadata_name = sys.argv[2]
meta_index = int(sys.argv[3])
predix=sys.argv[4]# 1 for first 10 min; 2for second 10 min; 3 for last 10 min; 4 for all
scaling=float(sys.argv[5])

#
# fileplace = 'U:\\rec\\video_track\\VT_self\\wt_pattern\\cs\\20220521_cs\\VT0521_480p/'
# metadata_name = 'metadata.csv'
# meta_index = 4
# predix = '0'  # 1 for 10 min; 2for 20 min; 3 for 30 min;4 for 40 min;5 for 50 min; 0 for all
# scaling = 1.6

fileplace_analysis = fileplace + 'analysis' + predix + '/'
if not os.path.exists(fileplace_analysis):  # 如果文件目录不存在则创建目录
    os.makedirs(fileplace_analysis)
import configparser

conf = configparser.ConfigParser()
conf.read('config.ini')
real_r = float(conf.get('Fixed_para', 'diameter')) / 2
# drosophila size F ~60pixel M ~50pixel
sensing_area = float(conf.get('Adjustable_para', 'sensing_area'))  # in pixel
scale_to_interaction = float(conf.get('Adjustable_para', 'scale_to_interaction'))
ssi_bin = float(conf.get('Adjustable_para', 'ssi_bin'))

metadata = pd.read_csv(fileplace + metadata_name, header=0)
fps = int(conf.get('Fixed_para', 'fps'))
csv_name = metadata.loc[meta_index, 'csv']
v_name = metadata.loc[meta_index, 'videoname']
x, y, r = metadata.loc[meta_index, 'x'], metadata.loc[meta_index, 'y'], metadata.loc[meta_index, 'r']


cycle_area = np.pi * r * r
n_inds = int(float(metadata.loc[meta_index, 'num']))
network_size = float(conf.get('Adjustable_para', 'network_size'))
network_size=math.ceil(n_inds*(n_inds-1)/2*network_size)

a=int(float(conf.get('Fixed_para', 'time_periods')))
step=float(conf.get('Fixed_para', 'time_windows'))
scaling_to_mm = real_r / r
# @timeout_decorator.timeout(max_timeout)
# def get_sigma_omega(G):
#     sigma = np.nan
#     omega =np.nan
#     try:
#         sigma = nx.sigma(G)
#     except:
#         sigma = np.nan
#     try:
#         omega = nx.omega(G)
#     except:
#         omega = np.nan
#     return sigma,omega
whole_ind_df = pd.DataFrame(columns={'videoname', 'id', 'xlabel', 'ylabel','timepoint',
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

g_df = pd.DataFrame(columns={'videoname','timepoint',
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


if n_inds > 1:
    # scaling = 1.6  # 1.6 for 480p to 768p.  real_r/r for pixel to mm

    for ts in range(a):
        l_list = []

        s1 = ts * step
        s2 = (ts + 2) * step
        s3 = (ts + 1) * step

        pos = pd.read_csv(fileplace + csv_name, header=0, index_col=None)
        moveinfo = pd.read_csv(fileplace_analysis + v_name + '_ind_moveinfo.csv', header=0, index_col=None)
        l_list = list(moveinfo.columns)
        pos = pd.merge(pos, moveinfo, on='position', how='inner')

        pos = pos[pos['position'] <= (s2 * 60 * fps)]
        if len(pos) == 0:
            continue
        pos = pos[pos['position'] > (s1 * 60 * fps)]
        if len(pos) == 0:
            continue
        pos.sort_values(by='position', inplace=True)
        pos.reset_index(drop=True, inplace=True)
        pos['position'] = pos['position'] - pos.loc[0, 'position']
        xlabel_list = []
        ylabel_list = []


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

        g_df.loc[ts, 'videoname'] = v_name
        g_df.loc[ts, 'timepoint'] = s3
        g_df.loc[ts, 'degree_assortativity_coefficient'] = []
        g_df.loc[ts, 'clustering_coefficient'] = []
        g_df.loc[ts, 'betweenness_centrality'] = 0
        g_df.loc[ts, 'diameter'] = []
        # g_df.loc[0, 'sigma'] = []
        # g_df.loc[0, 'omega'] = []
        g_df.loc[ts, 'degree'] = []
        g_df.loc[ts, 'global_efficiency'] = []

        ind_df = pd.DataFrame(columns={'videoname', 'id', 'xlabel', 'ylabel','timepoint',
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
        # total_grouped.to_csv(fileplace_analysis + v_name + '_interaction_bytimeline.csv')
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
                        g_df.loc[ts, 'global_efficiency'].append(ge)
                    if dac is not None and (not np.isnan(dac)):
                        g_df.loc[ts, 'degree_assortativity_coefficient'].append(dac)
                    # sigma = np.nan
                    # omega =np.nan
                    # if sys!='Windows':
                    #     try:
                    #         sigma,omega=get_sigma_omega(G)
                    #     except:
                    #         pass
                    if acc is not None and (not np.isnan(acc)):
                        g_df.loc[ts, 'clustering_coefficient'].append(acc)
                    if dg is not None and (not np.isnan(dg)):
                        g_df.loc[ts, 'diameter'].append(dg)
                    # if sigma is not None and (not np.isnan(sigma)):
                    #     g_df.loc[0, 'sigma'].append(sigma)
                    # if omega is not None and (not np.isnan(omega)):
                    #     g_df.loc[0, 'omega'].append(omega)
                    # print(sigma,omega)
                    count = count + 1
                except:
                    er2=er2+1
                    pass

            # print(v_name, i,'timepoint',s3, 'th dro. social network topological analysis finished. Total: ',
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
            g_df.loc[ts, 'degree'] = np.nanmean(ind_df['degree'])
        else:
            g_df.loc[ts, 'degree'] =np.nan
        if sum(~np.isnan(list(ind_df['betweenness_centrality'])))>0:
            g_df.loc[ts, 'betweenness_centrality'] = np.nanmean(ind_df['betweenness_centrality'])
        else:
            g_df.loc[ts, 'betweenness_centrality'] =np.nan
        if len(g_df.loc[ts, 'degree_assortativity_coefficient'])>0:
            g_df.loc[ts, 'degree_assortativity_coefficient'] = np.nanmean(g_df.loc[ts, 'degree_assortativity_coefficient'])
        else:
            g_df.loc[ts, 'degree_assortativity_coefficient'] =np.nan
        if len(g_df.loc[ts, 'clustering_coefficient'])>0:
            g_df.loc[ts, 'clustering_coefficient'] = np.nanmean(g_df.loc[ts, 'clustering_coefficient'])
        else:
            g_df.loc[ts, 'clustering_coefficient']=np.nan
        if len(g_df.loc[ts, 'diameter'])>0:
            g_df.loc[ts, 'diameter'] = np.nanmean(g_df.loc[ts, 'diameter'])
        else:
            g_df.loc[ts, 'diameter'] =np.nan
        if len(g_df.loc[ts, 'global_efficiency'])>0:
            g_df.loc[ts, 'global_efficiency'] = np.nanmean(g_df.loc[ts, 'global_efficiency'])
        else:
            g_df.loc[ts, 'global_efficiency'] =np.nan


        # if len(g_df.loc[0, 'sigma'])>0:
        #     g_df.loc[0, 'sigma'] = np.nanmean(g_df.loc[0, 'sigma'])
        # else:
        #     g_df.loc[0, 'sigma'] =np.nan
        # if len(g_df.loc[0, 'omega']):
        #     g_df.loc[0, 'omega'] = np.nanmean(g_df.loc[0, 'omega'])
        # else:
        #     g_df.loc[0, 'omega'] =np.nan
        if er+count>0:
            g_df.loc[ts, 'unconnected_social_network_proportion'] = er / (er + count)
        else:
            g_df.loc[ts, 'unconnected_social_network_proportion'] = np.nan
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
            print(v_name, i,'timepoint',s3, 'th dro. distance space analysis finished')
            frame_list = tmp_dis['position'] / fps / 60  # in minute
            distance_space = tmp_dis['min_val'] * scaling_to_mm
            plt.figure(figsize=(18, 6))
            plt.plot(frame_list, distance_space, alpha=0.5)
            plt.xlabel('Time(min)', fontsize=10)
            plt.ylabel('Distance(mm)', fontsize=10)
            plt.title(xlabel + ylabel + ' Distance from other drosophila')
            plt.tight_layout()
            plt.savefig(fileplace_analysis + v_name + '_' + xlabel + '_' + ylabel + '_social_space.png')
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
            print(v_name, i,'timepoint',s3, 'th dro. interaction frequency analysis finished')

            # i_interaction = tmp_interaction[tmp_interaction['total'] != 0]
            # i_interaction = i_interaction[['position', 'total']]
            # # i_interaction.to_csv(fileplace_analysis + v_name+xlabel+ylabel+'_ind_interaction.csv')
            # # interaction_columns = tmp_interaction.columns
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
            print(v_name, i,'timepoint',s3, 'th dro. acquaintances analysis finished')
            ind_df['timepoint']=s3
        whole_ind_df=pd.concat([ind_df,whole_ind_df])
        if sum(~np.isnan(list(ind_df['acquaintances'])))>0:
            g_df.loc[ts, 'acquaintances'] = np.nanmean(list(ind_df['acquaintances']))
        else:
            g_df.loc[ts, 'acquaintances'] =np.nan
        if sum(~np.isnan(list(ind_df['distance_space'])))>0:
            g_df.loc[ts, 'distance_space'] = np.nanmean(list(ind_df['distance_space']))
        else:
            g_df.loc[ts, 'distance_space'] =np.nan
        if sum(~np.isnan(list(ind_df['distance_space_at_edge'])))>0:
            g_df.loc[ts, 'distance_space_at_edge'] = np.nanmean(list(ind_df['distance_space_at_edge']))
        else:
            g_df.loc[ts, 'distance_space_at_edge']=np.nan
        if sum(~np.isnan(list(ind_df['distance_space_at_center'])))>0:
            g_df.loc[ts, 'distance_space_at_center'] = np.nanmean(list(ind_df['distance_space_at_center']))
        else:
            g_df.loc[ts, 'distance_space_at_center'] =np.nan
        if sum(~np.isnan(list(ind_df['distance_space_at_move'])))>0:
            g_df.loc[ts, 'distance_space_at_move'] = np.nanmean(list(ind_df['distance_space_at_move']))
        else:
            g_df.loc[ts, 'distance_space_at_move'] =np.nan
        if sum(~np.isnan(list(ind_df['distance_space_at_stop'])))>0:
            g_df.loc[ts, 'distance_space_at_stop'] = np.nanmean(list(ind_df['distance_space_at_stop']))
        else:
            g_df.loc[ts, 'distance_space_at_stop'] =np.nan
        if sum(~np.isnan(list(ind_df['SSI'].astype(float))))>0:
            # print(ind_df['SSI'])
            g_df.loc[ts, 'SSI'] = np.nanmean(ind_df['SSI'].astype(float))
        else:
            g_df.loc[ts, 'SSI'] = np.nan
        if sum(~np.isnan(list(ind_df['SSI_at_edge'].astype(float))))>0:
            g_df.loc[ts, 'SSI_at_edge'] = np.nanmean(ind_df['SSI_at_edge'].astype(float))
        else:
            g_df.loc[ts, 'SSI_at_edge'] = np.nan
        if sum(~np.isnan(list(ind_df['SSI_at_center'].astype(float))))>0:
            g_df.loc[ts, 'SSI_at_center'] = np.nanmean(ind_df['SSI_at_center'].astype(float))
        else:
            g_df.loc[ts, 'SSI_at_center'] = np.nan
        if sum(~np.isnan(list(ind_df['SSI_at_move'].astype(float))))>0:
            g_df.loc[ts, 'SSI_at_move'] = np.nanmean(ind_df['SSI_at_move'].astype(float))
        else:
            g_df.loc[ts, 'SSI_at_move'] = np.nan
        if sum(~np.isnan(list(ind_df['SSI_at_stop'].astype(float))))>0:
            g_df.loc[ts, 'SSI_at_stop'] = np.nanmean(ind_df['SSI_at_stop'].astype(float))
        else:
            g_df.loc[ts, 'SSI_at_stop'] = np.nan
        if sum(~np.isnan(list(ind_df['interaction'].astype(float))))>0:
            g_df.loc[ts, 'interaction'] = np.nanmean(ind_df['interaction'].astype(float))
        else:
            g_df.loc[ts, 'interaction'] = np.nan
        if sum(~np.isnan(list(ind_df['interaction_at_edge'].astype(float))))>0:
            g_df.loc[ts, 'interaction_at_edge'] = np.nanmean(ind_df['interaction_at_edge'].astype(float))
        else:
            g_df.loc[ts, 'interaction_at_edge'] = np.nan
        if sum(~np.isnan(list(ind_df['interaction_at_center'].astype(float))))>0:
            g_df.loc[ts, 'interaction_at_center'] = np.nanmean(ind_df['interaction_at_center'].astype(float))
        else:
            g_df.loc[ts, 'interaction_at_center'] = np.nan
        if sum(~np.isnan(list(ind_df['interaction_at_edge_proportion'].astype(float))))>0:
            g_df.loc[ts, 'interaction_at_edge_proportion'] = np.nanmean(ind_df['interaction_at_edge_proportion'].astype(float))
        else:
            g_df.loc[ts, 'interaction_at_edge_proportion'] = np.nan
        if sum(~np.isnan(list(ind_df['interaction_at_center_proportion'].astype(float))))>0:
            g_df.loc[ts, 'interaction_at_center_proportion'] = np.nanmean(ind_df['interaction_at_center_proportion'].astype(float))
        else:
            g_df.loc[ts, 'interaction_at_center_proportion'] = np.nan
        if sum(~np.isnan(list(ind_df['interaction_at_move'].astype(float))))>0:
            g_df.loc[ts, 'interaction_at_move'] = np.nanmean(ind_df['interaction_at_move'].astype(float))
        else:
            g_df.loc[ts, 'interaction_at_move'] = np.nan
        if sum(~np.isnan(list(ind_df['interaction_at_stop'].astype(float))))>0:
            g_df.loc[ts, 'interaction_at_stop'] = np.nanmean(ind_df['interaction_at_stop'].astype(float))
        else:
            g_df.loc[ts, 'interaction_at_stop'] = np.nan
        if sum(~np.isnan(list(ind_df['interaction_at_move_proportion'].astype(float))))>0:
            g_df.loc[ts, 'interaction_at_move_proportion'] = np.nanmean(ind_df['interaction_at_move_proportion'].astype(float))
        else:
            g_df.loc[ts, 'interaction_at_move_proportion'] = np.nan
        if sum(~np.isnan(list(ind_df['interaction_at_stop_proportion'].astype(float))))>0:
            g_df.loc[ts, 'interaction_at_stop_proportion'] = np.nanmean(ind_df['interaction_at_stop_proportion'].astype(float))
        else:
            g_df.loc[ts, 'interaction_at_stop_proportion'] = np.nan
        if sum(~np.isnan(list(ind_df['interaction_at_long_stop'].astype(float))))>0:
            g_df.loc[ts, 'interaction_at_long_stop'] = np.nanmean(ind_df['interaction_at_long_stop'].astype(float))
        else:
            g_df.loc[ts, 'interaction_at_long_stop'] = np.nan

        if sum(~np.isnan(list(ind_df['interaction_at_long_stop_proportion'].astype(float))))>0:
            g_df.loc[ts, 'interaction_at_long_stop_proportion'] = np.nanmean(ind_df['interaction_at_long_stop_proportion'].astype(float))
        else:
            g_df.loc[ts, 'interaction_at_long_stop_proportion'] = np.nan

        if sum(~np.isnan(list(ind_df['interaction_counts'].astype(float))))>0:
            g_df.loc[ts, 'interaction_counts'] = np.nanmean(ind_df['interaction_counts'].astype(float))
        else:
            g_df.loc[ts, 'interaction_counts'] = np.nan
        if sum(~np.isnan(list(ind_df['interaction_duration'].astype(float))))>0:
            g_df.loc[ts, 'interaction_duration'] = np.nanmean(ind_df['interaction_duration'].astype(float))
        else:
            g_df.loc[ts, 'interaction_duration'] = np.nan
        if sum(~np.isnan(list(ind_df['interaction_members'].astype(float))))>0:
            g_df.loc[ts, 'interaction_members'] = np.nanmean(ind_df['interaction_members'].astype(float))
        else:
            g_df.loc[ts, 'interaction_members'] = np.nan

    whole_ind_df_tocsv = whole_ind_df[['videoname', 'id', 'xlabel', 'ylabel','timepoint',
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
                     'interaction_at_edge_proportion',
                     'interaction_at_center',
                     'interaction_at_center_proportion',
                     'interaction_at_move',
                     'interaction_at_move_proportion',
                     'interaction_at_stop',
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
    whole_ind_df_tocsv.to_csv(fileplace_analysis + v_name + '_ind_interaction_sliced.csv')

    g_df = g_df[[
        'videoname','timepoint',
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
        'interaction_at_edge_proportion',
        'interaction_at_center',
        'interaction_at_center_proportion',
        'interaction_at_move',
        'interaction_at_move_proportion',
        'interaction_at_stop',
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
    g_df.to_csv(fileplace_analysis + v_name + '_avg_interaction_sliced.csv')
