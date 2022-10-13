import configparser
import sys
import numpy as np

import pandas as pd
import seaborn as sns
# custom_params = {"axes.spines.right": False, "axes.spines.top": False}
# sns.set_theme(style="ticks", rc=custom_params)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
#

conf=configparser.ConfigParser()
conf.read('config.ini')
radius=float(conf.get('Fixed_para','diameter'))/2
plot_factors=conf.get('Plot_para','plot_factors')
plot_type=conf.get('Plot_para','plot_type')
plot_swarm=conf.get('Plot_para','plot_swarm')
plot_by_video=conf.get('Plot_para','plot_by_video')
plot_with_logged_yscale=conf.get('Plot_para','plot_with_logged_yscale')

ci_value=int(float(conf.get('Plot_para','ci')))

if ',' in plot_factors:
    plot_factors=plot_factors.split(',')
else:
    plot_factors=[plot_factors]
#
fileplace=sys.argv[1]
predix=sys.argv[2]
# fileplace='U:\\rec\\video_track\\VT_self\\2mm3mm\\20220122_2mm_3mm\\VT0122_480p/'
# predix='0'
fileplace_analysis=fileplace+'analysis'+predix+'/'

motion_df_filename='ind_motion_sliced.csv'
interaction_df_filename='ind_interaction_sliced.csv'
if plot_by_video=='1':
    motion_df_filename='avg_motion_sliced.csv'
    interaction_df_filename='avg_interaction_sliced.csv'
motion_df=pd.read_csv(fileplace_analysis+motion_df_filename,header=0,)
interaction_df=pd.read_csv(fileplace_analysis+interaction_df_filename,header=0,)
plot_factor_num=0
pf_list=[]
pf_factors=[]
for i in range(len(plot_factors),0,-1):
    f=plot_factors[i-1]
    fd=list(set(list(motion_df.loc[~motion_df[f].isna(),f])))
    if len(fd)>1:
        fd=np.sort(fd)
        pf_list.append(f)
        pf_factors.append(fd)
pass
if len(pf_list)>3:
    pf_list=pf_list[0:3]
pf_len=len(pf_list)
motion_image=fileplace_analysis+'motion_analysis_sliced_plot.pdf'
interaction_image=fileplace_analysis+'interaction_analysis_sliced_plot.pdf'

if pf_len==0:
    pf_list=[plot_factors[len(plot_factors)-1]]
    pf_factors=[list(set(list(motion_df.loc[~motion_df[pf_list[0]].isna(),pf_list[0]])))]

if pf_len>=2:
    # search_area_time	search_area	search_area_unit_move
    with PdfPages(motion_image) as pdf:
        sns.set()
        plt.figure(figsize = (40, 32))
        # plt.rcParams['figure.constrained_layout.use'] = True
        # move length
        ax1=plt.subplot2grid((5,3),(0,0),colspan=1,rowspan=1)
        ax2=plt.subplot2grid((5,3),(0,1),colspan=1,rowspan=1)
        ax3=plt.subplot2grid((5,3),(0,2),colspan=1,rowspan=1)
        # Velocity
        ax4=plt.subplot2grid((5,3),(1,0),colspan=1,rowspan=1)
        ax5=plt.subplot2grid((5,3),(1,1),colspan=1,rowspan=1)
        ax6=plt.subplot2grid((5,3),(1,2),colspan=1,rowspan=1)

        # ax4=plt.subplot2grid((3,12),(1,0),colspan=3,rowspan=1)
        # ax5=plt.subplot2grid((3,12),(1,3),colspan=3,rowspan=1)
        # ax6=plt.subplot2grid((3,12),(1,6),colspan=3,rowspan=1)
        # ax7=plt.subplot2grid((3,12),(1,9),colspan=3,rowspan=1)
        # max Velocity
        ax8=plt.subplot2grid((5,3),(2,0),colspan=1,rowspan=1)
        ax9=plt.subplot2grid((5,3),(2,1),colspan=1,rowspan=1)
        ax10=plt.subplot2grid((5,3),(2,2),colspan=1,rowspan=1)
        # move time
        ax11=plt.subplot2grid((5,3),(3,0),colspan=1,rowspan=1)
        ax12=plt.subplot2grid((5,3),(3,1),colspan=1,rowspan=1)
        ax13=plt.subplot2grid((5,3),(3,2),colspan=1,rowspan=1)
        # redge
        ax14=plt.subplot2grid((5,3),(4,0),colspan=1,rowspan=1)
        ax15=plt.subplot2grid((5,3),(4,1),colspan=1,rowspan=1)
        ax16 = plt.subplot2grid((5, 3), (4, 2), colspan=1, rowspan=1)


        motion_df['r_edge']=100*motion_df['r_edge']
        motion_df['r_dist']=100*motion_df['r_dist']
        motion_df['movelength_ratio_at_edge'] = 100 * motion_df['movelength_ratio_at_edge']
        motion_df['move_time_threshed']=100*motion_df['move_time_threshed']
        motion_df['move_proportion_at_edge']=100*motion_df['move_proportion_at_edge']
        motion_df['move_proportion_at_center']=100*motion_df['move_proportion_at_center']

        sns.lineplot(x="timepoint", y="total_move_length_threshed", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[1], style_order=pf_factors[1], data=motion_df, ci=ci_value,ax=ax1)
        sns.lineplot(x="timepoint", y="movelength_at_edge", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[1],
                     style_order=pf_factors[1], data=motion_df, ci=ci_value, ax=ax2)
        sns.lineplot(x="timepoint", y="movelength_at_center", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[1],
                     style_order=pf_factors[1], data=motion_df, ci=ci_value, ax=ax3)

        sns.lineplot(x="timepoint", y="velocity_threshed", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[1],
                     style_order=pf_factors[1], data=motion_df, ci=ci_value, ax=ax4)
        sns.lineplot(x="timepoint", y="velocity_at_edge", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[1],
                     style_order=pf_factors[1], data=motion_df, ci=ci_value, ax=ax5)
        sns.lineplot(x="timepoint", y="velocity_at_center", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[1],
                     style_order=pf_factors[1], data=motion_df, ci=ci_value, ax=ax6)

        sns.lineplot(x="timepoint", y="max_velocity_threshed", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[1],
                     style_order=pf_factors[1], data=motion_df, ci=ci_value, ax=ax8)
        sns.lineplot(x="timepoint", y="max_velocity_at_edge", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[1],
                     style_order=pf_factors[1], data=motion_df, ci=ci_value, ax=ax9)
        sns.lineplot(x="timepoint", y="max_velocity_at_center", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[1],
                     style_order=pf_factors[1], data=motion_df, ci=ci_value, ax=ax10)

        sns.lineplot(x="timepoint", y="move_time_threshed", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[1],
                     style_order=pf_factors[1], data=motion_df, ci=ci_value, ax=ax11)
        sns.lineplot(x="timepoint", y="move_proportion_at_edge", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[1],
                     style_order=pf_factors[1], data=motion_df, ci=ci_value, ax=ax12)
        sns.lineplot(x="timepoint", y="move_proportion_at_center", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[1],
                     style_order=pf_factors[1], data=motion_df, ci=ci_value, ax=ax13)

        sns.lineplot(x="timepoint", y="r_edge", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[1],
                     style_order=pf_factors[1], data=motion_df, ci=ci_value, ax=ax14)
        sns.lineplot(x="timepoint", y="r_dist", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[1],
                     style_order=pf_factors[1], data=motion_df, ci=ci_value, ax=ax15)
        sns.lineplot(x="timepoint", y="movelength_ratio_at_edge", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[1],
                     style_order=pf_factors[1], data=motion_df, ci=ci_value, ax=ax16)

        ax2.set_title(f'Locomotion', fontsize=14, fontweight='bold',  color='#30302f', loc='center')
        ax1.set( ylabel="Total move length(mm)")
        ax2.set( ylabel="Move length at arena edge(mm)")
        ax3.set( ylabel="Move length at arena centre(mm)")

        ax5.set_title(f'Velocity', fontsize=14, fontweight='bold',  color='#30302f', loc='center')

        ax4.set( ylabel="Average velocity(mm/s)")
        ax5.set( ylabel="Average velocity at arena edge(mm/s)")
        ax6.set( ylabel="Average velocity at arena centre(mm/s)")

        ax8.set( ylabel="Maximum velocity(mm/s)")
        ax9.set( ylabel="Maximum velocity at arena edge(mm/s)")
        ax10.set( ylabel="Maximum velocity at arena centre(mm/s)")

        ax12.set_title(f'Move time', fontsize=14, fontweight='bold',  color='#30302f', loc='center')
        ax11.set( ylabel="Move time proportion(%)")
        ax12.set( ylabel="Move time proportion at arena edge(%)")
        ax13.set( ylabel="Move time proportion at arena centre(%)")

        ax15.set_title(f'Spatial preference', fontsize=14, fontweight='bold',  color='#30302f', loc='center')
        ax14.set( ylabel="Time proportion spent at edge(%)")
        ax15.set( ylabel="Average distance from the arena center/Radius(%)")
        ax16.set(ylabel="Move lenngth ratio at edge(%)")

        plt.tight_layout()
        pdf.savefig()
        plt.close()
        #page 2
        sns.set()
        plt.figure(figsize = (40, 26))
        # plt.rcParams['figure.constrained_layout.use'] = True
        ax1=plt.subplot2grid((4,12),(0,0),colspan=4,rowspan=1)
        ax2=plt.subplot2grid((4,12),(0,4),colspan=4,rowspan=1)
        ax3=plt.subplot2grid((4,12),(0,8),colspan=4,rowspan=1)
        ax4=plt.subplot2grid((4,12),(1,0),colspan=4,rowspan=1)
        ax5=plt.subplot2grid((4,12),(1,4),colspan=4,rowspan=1)
        ax6=plt.subplot2grid((4,12),(1,8),colspan=4,rowspan=1)
        ax7=plt.subplot2grid((4,12),(2,0),colspan=4,rowspan=1)
        ax8=plt.subplot2grid((4,12),(2,4),colspan=4,rowspan=1)
        ax9=plt.subplot2grid((4,12),(2,8),colspan=4,rowspan=1)
        ax10=plt.subplot2grid((4,12),(3,0),colspan=4,rowspan=1)
        ax11=plt.subplot2grid((4,12),(3,4),colspan=4,rowspan=1)
        ax12=plt.subplot2grid((4,12),(3,8),colspan=4,rowspan=1)
        motion_df['search_area_time']=100*motion_df['search_area_time']
        sns.lineplot(x="timepoint", y="tracks_num", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[1],
                     style_order=pf_factors[1], data=motion_df, ci=ci_value, ax=ax1)
        sns.lineplot(x="timepoint", y="tracks_duration", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[1],
                     style_order=pf_factors[1], data=motion_df, ci=ci_value, ax=ax2)
        sns.lineplot(x="timepoint", y="tracks_length", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[1],
                     style_order=pf_factors[1], data=motion_df, ci=ci_value, ax=ax3)

        sns.lineplot(x="timepoint", y="stop_duration", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[1],
                     style_order=pf_factors[1], data=motion_df, ci=ci_value, ax=ax4)
        sns.lineplot(x="timepoint", y="long_stop_num", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[1],
                     style_order=pf_factors[1], data=motion_df, ci=ci_value, ax=ax5)
        sns.lineplot(x="timepoint", y="long_stop_duration", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[1],
                     style_order=pf_factors[1], data=motion_df, ci=ci_value, ax=ax6)

        sns.lineplot(x="timepoint", y="search_area_time", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[1],
                     style_order=pf_factors[1], data=motion_df, ci=ci_value, ax=ax7)
        sns.lineplot(x="timepoint", y="search_area", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[1],
                     style_order=pf_factors[1], data=motion_df, ci=ci_value, ax=ax8)
        sns.lineplot(x="timepoint", y="search_area_unit_move", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[1],
                     style_order=pf_factors[1], data=motion_df, ci=ci_value, ax=ax9)

        sns.lineplot(x="timepoint", y="track_straightness", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[1],
                     style_order=pf_factors[1], data=motion_df, ci=ci_value, ax=ax10)
        sns.lineplot(x="timepoint", y="track_straightness_at_edge", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[1],
                     style_order=pf_factors[1], data=motion_df, ci=ci_value, ax=ax11)
        sns.lineplot(x="timepoint", y="track_straightness_non_edge", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[1],
                     style_order=pf_factors[1], data=motion_df, ci=ci_value, ax=ax12)

        ax2.set_title(f'Tracks', fontsize=14, fontweight='bold',  color='#30302f', loc='center')
        ax1.set( ylabel="Tracks number")
        ax2.set( ylabel="Average track duration(s)")
        ax3.set( ylabel="Average track length(mm)")

        ax5.set_title(f'Inactivity episodes', fontsize=14, fontweight='bold',  color='#30302f', loc='center')
        ax4.set( ylabel="Average inactivity duration(s)")
        ax5.set( ylabel="Long inactivity episodes number")
        ax6.set( ylabel="Average long inactivity episodes duration(s)")

        ax8.set_title(f'Arena exploration', fontsize=14, fontweight='bold',  color='#30302f', loc='center')
        ax7.set( ylabel="Time used to search area/Recorded time(%)")
        ax8.set( ylabel="Arena explored proportion at a given time")
        ax9.set( ylabel="Arena explored proportion/Travel length(/mm)")

        ax11.set_title(f'Track straightness', fontsize=14, fontweight='bold',  color='#30302f', loc='center')
        ax10.set( ylabel="Average track straightness")
        ax11.set( ylabel="Track straightness at arena edge")
        ax12.set( ylabel="Track straightness at arena centre")

        # if plot_with_logged_yscale=='1':
        #     ax1.set(yscale="log")
        #     ax2.set(yscale="log")
        #     ax3.set(yscale="log")
        #     ax4.set(yscale="log")
        #     ax5.set(yscale="log")
        #     ax6.set(yscale="log")
        #     ax7.set(yscale="log")
        #     ax8.set(yscale="log")
        #     ax9.set(yscale="log")
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        #page 3
        sns.set()
        # plt.figure()
        fig, axes = plt.subplots(nrows=4, ncols=3,figsize = (40, 26))
        sns.lineplot(x="timepoint", y="angular_velocity", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[1],
                     style_order=pf_factors[1], data=motion_df, ci=ci_value, ax=axes[0,0])
        sns.lineplot(x="timepoint", y="angular_velocity_at_edge", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[1],
                     style_order=pf_factors[1], data=motion_df, ci=ci_value, ax=axes[0,1])
        sns.lineplot(x="timepoint", y="angular_velocity_non_edge", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[1],
                     style_order=pf_factors[1], data=motion_df, ci=ci_value, ax=axes[0,2])

        sns.lineplot(x="timepoint", y="max_angular_velocity", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[1],
                     style_order=pf_factors[1], data=motion_df, ci=ci_value, ax=axes[1,0])
        sns.lineplot(x="timepoint", y="max_angular_velocity_at_edge", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[1],
                     style_order=pf_factors[1], data=motion_df, ci=ci_value, ax=axes[1,1])
        sns.lineplot(x="timepoint", y="max_angular_velocity_non_edge", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[1],
                     style_order=pf_factors[1], data=motion_df, ci=ci_value, ax=axes[1,2])

        sns.lineplot(x="timepoint", y="meander", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[1],
                     style_order=pf_factors[1], data=motion_df, ci=ci_value, ax=axes[2, 0])
        sns.lineplot(x="timepoint", y="meander_at_edge", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[1],
                     style_order=pf_factors[1], data=motion_df, ci=ci_value, ax=axes[2, 1])
        sns.lineplot(x="timepoint", y="meander_non_edge", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[1],
                     style_order=pf_factors[1], data=motion_df, ci=ci_value, ax=axes[2, 2])

        sns.lineplot(x="timepoint", y="max_meander", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[1],
                     style_order=pf_factors[1], data=motion_df, ci=ci_value, ax=axes[3, 0])
        sns.lineplot(x="timepoint", y="max_meander_at_edge", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[1],
                     style_order=pf_factors[1], data=motion_df, ci=ci_value, ax=axes[3, 1])
        sns.lineplot(x="timepoint", y="max_meander_non_edge", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[1],
                     style_order=pf_factors[1], data=motion_df, ci=ci_value, ax=axes[3, 2])


        axes[0,1].set_title(f'Average angular', fontsize=14, fontweight='bold',  color='#30302f', loc='center')
        axes[0,0].set( ylabel="Angular velocity(rad/s)")
        axes[0,1].set( ylabel="Angular velocity at arena edge(rad/s)")
        axes[0,2].set( ylabel="Angular velocity at arena centre(rad/s)")

        axes[1,1].set_title(f'Maximum angular', fontsize=14, fontweight='bold',  color='#30302f', loc='center')
        axes[1,0].set( ylabel="Maximum angular velocity(rad/s)")
        axes[1,1].set( ylabel="Maximum angular velocity at arena edge(rad/s)")
        axes[1,2].set( ylabel="Maximum angular velocity at arena centre(rad/s)")

        axes[2,1].set_title(f'Meander', fontsize=14, fontweight='bold',  color='#30302f', loc='center')
        axes[2,0].set( ylabel="Meander(rad/mm)")
        axes[2,1].set( ylabel="Meander at arena edge(rad/mm)")
        axes[2,2].set( ylabel="Meander at arena centre(rad/mm)")

        axes[3,1].set_title(f'Maximum meander', fontsize=14, fontweight='bold',  color='#30302f', loc='center')
        axes[3,0].set( ylabel="Maximum meander(rad/mm)")
        axes[3,1].set( ylabel="Maximum meander velocity at arena edge(rad/mm)")
        axes[3,2].set( ylabel="Maximum meander velocity at arena centre(rad/mm)")

        if plot_with_logged_yscale=='1':
            axes[2,0].set(yscale="log")
            axes[2,1].set(yscale="log")
            axes[2,2].set(yscale="log")
            axes[3,0].set(yscale="log")
            axes[3,1].set(yscale="log")
            axes[3,2].set(yscale="log")

        #     for i in range(4):
        #         for j in range(3):
        #             axes[i,j].set(yscale="log")
        plt.tight_layout()
        pdf.savefig()
        plt.close()
    with PdfPages(interaction_image) as pdf:
        #page 4
        sns.set()
        # plt.figure()
        fig, axes = plt.subplots(nrows=4, ncols=4,figsize = (40, 20))
        sns.lineplot(x="timepoint", y="distance_space", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[1],
                     style_order=pf_factors[1], data=interaction_df, ci=ci_value, ax=axes[0, 0])

        sns.lineplot(x="timepoint", y="distance_space_at_edge", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[1],
                     style_order=pf_factors[1], data=interaction_df, ci=ci_value, ax=axes[1, 0])
        sns.lineplot(x="timepoint", y="distance_space_at_center", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[1],
                     style_order=pf_factors[1], data=interaction_df, ci=ci_value, ax=axes[1, 1])
        sns.lineplot(x="timepoint", y="distance_space_at_move", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[1],
                     style_order=pf_factors[1], data=interaction_df, ci=ci_value, ax=axes[1, 2])
        sns.lineplot(x="timepoint", y="distance_space_at_stop", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[1],
                     style_order=pf_factors[1], data=interaction_df, ci=ci_value, ax=axes[1, 3])

        sns.lineplot(x="timepoint", y="SSI", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[1],
                     style_order=pf_factors[1], data=interaction_df, ci=ci_value, ax=axes[2, 0])

        sns.lineplot(x="timepoint", y="SSI_at_edge", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[1],
                     style_order=pf_factors[1], data=interaction_df, ci=ci_value, ax=axes[3, 0])
        sns.lineplot(x="timepoint", y="SSI_at_center", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[1],
                     style_order=pf_factors[1], data=interaction_df, ci=ci_value, ax=axes[3, 1])
        sns.lineplot(x="timepoint", y="SSI_at_move", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[1],
                     style_order=pf_factors[1], data=interaction_df, ci=ci_value, ax=axes[3, 2])
        sns.lineplot(x="timepoint", y="SSI_at_stop", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[1],
                     style_order=pf_factors[1], data=interaction_df, ci=ci_value, ax=axes[3, 3])


        axes[0,1].axis('off')
        axes[0,2].axis('off')
        axes[0,3].axis('off')
        axes[2,1].axis('off')
        axes[2,2].axis('off')
        axes[2,3].axis('off')
        axes[0,1].set_title(f'Social distance space', fontsize=14, fontweight='bold',  color='#30302f', loc='right')

        axes[0,0].set( ylabel="Distance space(mm)")

        axes[1,0].set( ylabel="Distance space at arena edge(mm)")
        axes[1,1].set( ylabel="Distance space at arena centre(mm)")
        axes[1,2].set( ylabel="Distance space at activity episodes(mm)")
        axes[1,3].set( ylabel="Distance space at inactivity episodes(mm)")

        axes[2,1].set_title(f'Social space index(SSI)', fontsize=14, fontweight='bold',  color='#30302f', loc='right')

        axes[3,0].set( ylabel="SSI at arena edge")
        axes[3,1].set( ylabel="SSI at arena centre")
        axes[3,2].set( ylabel="SSI at activity episodes")
        axes[3,3].set( ylabel="SSI at inactivity episodes")


# if plot_with_logged_yscale=='1':
        #     for i in range(4):
        #         for j in range(3):
        #             axes[i,j].set(yscale="log")
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        #page 5
        sns.set()
        fig, axes = plt.subplots(nrows=4, ncols=4,figsize = (40, 20))
        interaction_df['interaction']=100*interaction_df['interaction']
        interaction_df['interaction_at_edge_proportion']=100*interaction_df['interaction_at_edge_proportion']
        interaction_df['interaction_at_center_proportion']=100*interaction_df['interaction_at_center_proportion']
        interaction_df['interaction_at_edge']=100*interaction_df['interaction_at_edge']
        interaction_df['interaction_at_center']=100*interaction_df['interaction_at_center']

        interaction_df['interaction_at_move']=100*interaction_df['interaction_at_move']
        interaction_df['interaction_at_stop']=100*interaction_df['interaction_at_stop']
        interaction_df['interaction_at_move_proportion']=100*interaction_df['interaction_at_move_proportion']
        interaction_df['interaction_at_stop_proportion']=100*interaction_df['interaction_at_stop_proportion']

        interaction_df['interaction_at_long_stop']=100*interaction_df['interaction_at_long_stop']
        interaction_df['interaction_at_long_stop_proportion']=100*interaction_df['interaction_at_long_stop_proportion']

        sns.lineplot(x="timepoint", y="interaction", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[1],
                     style_order=pf_factors[1], data=interaction_df, ci=ci_value, ax=axes[0, 0])
        sns.lineplot(x="timepoint", y="interaction_counts", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[1],
                     style_order=pf_factors[1], data=interaction_df, ci=ci_value, ax=axes[0, 1])
        sns.lineplot(x="timepoint", y="interaction_duration", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[1],
                     style_order=pf_factors[1], data=interaction_df, ci=ci_value, ax=axes[0, 2])
        sns.lineplot(x="timepoint", y="interaction_members", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[1],
                     style_order=pf_factors[1], data=interaction_df, ci=ci_value, ax=axes[0, 3])

        sns.lineplot(x="timepoint", y="interaction_at_edge", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[1],
                     style_order=pf_factors[1], data=interaction_df, ci=ci_value, ax=axes[1, 0])
        sns.lineplot(x="timepoint", y="interaction_at_center", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[1],
                     style_order=pf_factors[1], data=interaction_df, ci=ci_value, ax=axes[1, 1])
        sns.lineplot(x="timepoint", y="interaction_at_edge_proportion", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[1],
                     style_order=pf_factors[1], data=interaction_df, ci=ci_value, ax=axes[1, 2])
        sns.lineplot(x="timepoint", y="interaction_at_center_proportion", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[1],
                     style_order=pf_factors[1], data=interaction_df, ci=ci_value, ax=axes[1, 3])

        sns.lineplot(x="timepoint", y="interaction_at_move", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[1],
                     style_order=pf_factors[1], data=interaction_df, ci=ci_value, ax=axes[2, 0])
        sns.lineplot(x="timepoint", y="interaction_at_stop", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[1],
                     style_order=pf_factors[1], data=interaction_df, ci=ci_value, ax=axes[2, 1])
        sns.lineplot(x="timepoint", y="interaction_at_move_proportion", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[1],
                     style_order=pf_factors[1], data=interaction_df, ci=ci_value, ax=axes[2, 2])
        sns.lineplot(x="timepoint", y="interaction_at_stop_proportion", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[1],
                     style_order=pf_factors[1], data=interaction_df, ci=ci_value, ax=axes[2, 3])

        sns.lineplot(x="timepoint", y="interaction_at_long_stop", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[1],
                     style_order=pf_factors[1], data=interaction_df, ci=ci_value, ax=axes[3, 0])
        sns.lineplot(x="timepoint", y="interaction_at_long_stop_proportion", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[1],
                     style_order=pf_factors[1], data=interaction_df, ci=ci_value, ax=axes[3, 1])



        # axes[3,1].axis('off')
        axes[3,2].axis('off')
        axes[3,3].axis('off')
        axes[0,1].set_title(f'Interaction', fontsize=14, fontweight='bold',  color='#30302f', loc='right')

        axes[0,0].set( ylabel="Total interaction duration(%)")
        axes[0,1].set( ylabel="Interaction episode counts")
        axes[0,2].set( ylabel="Interaction episode duration(s)")
        axes[0,3].set( ylabel="Average number of crowed drosophila at interaction")

        axes[1,0].set( ylabel="Total interaction duration at arena edge(%)")
        axes[1,1].set( ylabel="Total interaction duration at arena centre(%)")
        axes[1,2].set( ylabel="The time proportion for interaction at edge(%)")
        axes[1,3].set( ylabel="The time proportion for interaction  at centre(%)")

        axes[2,0].set( ylabel="Total interaction duration at activity episodes(%)")
        axes[2,1].set( ylabel="Total interaction duration at inactivity episodes(%)")
        axes[2,2].set( ylabel="The time proportion for interaction at activity episodes(%)")
        axes[2,3].set( ylabel="The time proportion for interaction at inactivity episodes(%)")

        axes[3,0].set( ylabel="Total interaction duration at long-stop(%)")
        axes[3,1].set( ylabel="The time proportion for interaction at long-stop(%)")

        plt.tight_layout()
        pdf.savefig()
        plt.close()
        #page 6
        sns.set()
        fig, axes = plt.subplots(nrows=2, ncols=4,figsize = (40, 14))

        if plot_by_video=='1':
            interaction_df['unconnected_social_network_proportion']=100*interaction_df['unconnected_social_network_proportion']
            sns.lineplot(x="timepoint", y="acquaintances", hue=pf_list[0],
                         hue_order=pf_factors[0],
                         style=pf_list[1],
                         style_order=pf_factors[1], data=interaction_df, ci=ci_value, ax=axes[0, 0])
            sns.lineplot(x="timepoint", y="degree", hue=pf_list[0],
                         hue_order=pf_factors[0],
                         style=pf_list[1],
                         style_order=pf_factors[1], data=interaction_df, ci=ci_value, ax=axes[0, 1])
            sns.lineplot(x="timepoint", y="diameter", hue=pf_list[0],
                         hue_order=pf_factors[0],
                         style=pf_list[1],
                         style_order=pf_factors[1], data=interaction_df, ci=ci_value, ax=axes[0, 2])
            sns.lineplot(x="timepoint", y="unconnected_social_network_proportion", hue=pf_list[0],
                         hue_order=pf_factors[0],
                         style=pf_list[1],
                         style_order=pf_factors[1], data=interaction_df, ci=ci_value, ax=axes[0, 3])

            sns.lineplot(x="timepoint", y="degree_assortativity_coefficient", hue=pf_list[0],
                         hue_order=pf_factors[0],
                         style=pf_list[1],
                         style_order=pf_factors[1], data=interaction_df, ci=ci_value, ax=axes[1, 0])
            sns.lineplot(x="timepoint", y="clustering_coefficient", hue=pf_list[0],
                         hue_order=pf_factors[0],
                         style=pf_list[1],
                         style_order=pf_factors[1], data=interaction_df, ci=ci_value, ax=axes[1, 1])
            sns.lineplot(x="timepoint", y="betweenness_centrality", hue=pf_list[0],
                         hue_order=pf_factors[0],
                         style=pf_list[1],
                         style_order=pf_factors[1], data=interaction_df, ci=ci_value, ax=axes[1, 2])
            sns.lineplot(x="timepoint", y="global_efficiency", hue=pf_list[0],
                         hue_order=pf_factors[0],
                         style=pf_list[1],
                         style_order=pf_factors[1], data=interaction_df, ci=ci_value, ax=axes[1, 3])

        else:
            sns.lineplot(x="timepoint", y="acquaintances", hue=pf_list[0],
                         hue_order=pf_factors[0],
                         style=pf_list[1],
                         style_order=pf_factors[1], data=interaction_df, ci=ci_value, ax=axes[0, 0])
            sns.lineplot(x="timepoint", y="degree", hue=pf_list[0],
                         hue_order=pf_factors[0],
                         style=pf_list[1],
                         style_order=pf_factors[1], data=interaction_df, ci=ci_value, ax=axes[0, 1])
            sns.lineplot(x="timepoint", y="dominating", hue=pf_list[0],
                         hue_order=pf_factors[0],
                         style=pf_list[1],
                         style_order=pf_factors[1], data=interaction_df, ci=ci_value, ax=axes[0, 2])

            sns.lineplot(x="timepoint", y="closeness_centrality", hue=pf_list[0],
                         hue_order=pf_factors[0],
                         style=pf_list[1],
                         style_order=pf_factors[1], data=interaction_df, ci=ci_value, ax=axes[1, 0])
            sns.lineplot(x="timepoint", y="clustering_coefficient", hue=pf_list[0],
                         hue_order=pf_factors[0],
                         style=pf_list[1],
                         style_order=pf_factors[1], data=interaction_df, ci=ci_value, ax=axes[1, 1])
            sns.lineplot(x="timepoint", y="betweenness_centrality", hue=pf_list[0],
                         hue_order=pf_factors[0],
                         style=pf_list[1],
                         style_order=pf_factors[1], data=interaction_df, ci=ci_value, ax=axes[1, 2])
            sns.lineplot(x="timepoint", y="eccentricity", hue=pf_list[0],
                         hue_order=pf_factors[0],
                         style=pf_list[1],
                         style_order=pf_factors[1], data=interaction_df, ci=ci_value, ax=axes[1, 3])



        axes[0,1].set_title(f'Social networks', fontsize=14, fontweight='bold',  color='#30302f', loc='right')
        if plot_by_video=='1':
            axes[0,0].set( ylabel="Acquaintances")
            axes[0,1].set( ylabel="Network degree")
            axes[0,2].set( ylabel="Network diameter")
            axes[0,3].set( ylabel="Unconnected social network proportion(%)")

            axes[1,0].set( ylabel="Degree assortativity coefficient")
            axes[1,1].set( ylabel="Clustering coefficient")
            axes[1,2].set( ylabel="Betweenness centrality")
            axes[1,3].set( ylabel="Global efficiency")
        else:
            axes[0,0].set( ylabel="Acquaintances")
            axes[0,1].set( ylabel="Network degree")
            axes[0,2].set( ylabel="Dominating")
            axes[0, 3].axis('off')
            axes[1,0].set( ylabel="Closeness centrality")
            axes[1,1].set( ylabel="Clustering coefficient")
            axes[1,2].set( ylabel="Betweenness centrality")
            axes[1,3].set( ylabel="Eccentricity")
        plt.tight_layout()
        pdf.savefig()
        plt.close()

if pf_len<2:
    # search_area_time	search_area	search_area_unit_move
    with PdfPages(motion_image) as pdf:
        sns.set()
        plt.figure(figsize=(40, 32))
        # plt.rcParams['figure.constrained_layout.use'] = True
        # move length
        ax1 = plt.subplot2grid((5, 3), (0, 0), colspan=1, rowspan=1)
        ax2 = plt.subplot2grid((5, 3), (0, 1), colspan=1, rowspan=1)
        ax3 = plt.subplot2grid((5, 3), (0, 2), colspan=1, rowspan=1)
        # Velocity
        ax4 = plt.subplot2grid((5, 3), (1, 0), colspan=1, rowspan=1)
        ax5 = plt.subplot2grid((5, 3), (1, 1), colspan=1, rowspan=1)
        ax6 = plt.subplot2grid((5, 3), (1, 2), colspan=1, rowspan=1)

        # ax4=plt.subplot2grid((3,12),(1,0),colspan=3,rowspan=1)
        # ax5=plt.subplot2grid((3,12),(1,3),colspan=3,rowspan=1)
        # ax6=plt.subplot2grid((3,12),(1,6),colspan=3,rowspan=1)
        # ax7=plt.subplot2grid((3,12),(1,9),colspan=3,rowspan=1)
        # max Velocity
        ax8 = plt.subplot2grid((5, 3), (2, 0), colspan=1, rowspan=1)
        ax9 = plt.subplot2grid((5, 3), (2, 1), colspan=1, rowspan=1)
        ax10 = plt.subplot2grid((5, 3), (2, 2), colspan=1, rowspan=1)
        # move time
        ax11 = plt.subplot2grid((5, 3), (3, 0), colspan=1, rowspan=1)
        ax12 = plt.subplot2grid((5, 3), (3, 1), colspan=1, rowspan=1)
        ax13 = plt.subplot2grid((5, 3), (3, 2), colspan=1, rowspan=1)
        # redge
        ax14 = plt.subplot2grid((5, 3), (4, 0), colspan=1, rowspan=1)
        ax15 = plt.subplot2grid((5, 3), (4, 1), colspan=1, rowspan=1)
        ax16 = plt.subplot2grid((5, 3), (4, 2), colspan=1, rowspan=1)

        motion_df['r_edge'] = 100 * motion_df['r_edge']
        motion_df['r_dist'] = 100 * motion_df['r_dist']
        motion_df['movelength_ratio_at_edge'] = 100 * motion_df['movelength_ratio_at_edge']
        motion_df['move_time_threshed'] = 100 * motion_df['move_time_threshed']
        motion_df['move_proportion_at_edge'] = 100 * motion_df['move_proportion_at_edge']
        motion_df['move_proportion_at_center'] = 100 * motion_df['move_proportion_at_center']

        sns.lineplot(x="timepoint", y="total_move_length_threshed", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[0], style_order=pf_factors[0], data=motion_df, ci=ci_value, ax=ax1)
        sns.lineplot(x="timepoint", y="movelength_at_edge", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[0],
                     style_order=pf_factors[0], data=motion_df, ci=ci_value, ax=ax2)
        sns.lineplot(x="timepoint", y="movelength_at_center", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[0],
                     style_order=pf_factors[0], data=motion_df, ci=ci_value, ax=ax3)

        sns.lineplot(x="timepoint", y="velocity_threshed", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[0],
                     style_order=pf_factors[0], data=motion_df, ci=ci_value, ax=ax4)
        sns.lineplot(x="timepoint", y="velocity_at_edge", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[0],
                     style_order=pf_factors[0], data=motion_df, ci=ci_value, ax=ax5)
        sns.lineplot(x="timepoint", y="velocity_at_center", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[0],
                     style_order=pf_factors[0], data=motion_df, ci=ci_value, ax=ax6)

        sns.lineplot(x="timepoint", y="max_velocity_threshed", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[0],
                     style_order=pf_factors[0], data=motion_df, ci=ci_value, ax=ax8)
        sns.lineplot(x="timepoint", y="max_velocity_at_edge", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[0],
                     style_order=pf_factors[0], data=motion_df, ci=ci_value, ax=ax9)
        sns.lineplot(x="timepoint", y="max_velocity_at_center", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[0],
                     style_order=pf_factors[0], data=motion_df, ci=ci_value, ax=ax10)

        sns.lineplot(x="timepoint", y="move_time_threshed", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[0],
                     style_order=pf_factors[0], data=motion_df, ci=ci_value, ax=ax11)
        sns.lineplot(x="timepoint", y="move_proportion_at_edge", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[0],
                     style_order=pf_factors[0], data=motion_df, ci=ci_value, ax=ax12)
        sns.lineplot(x="timepoint", y="move_proportion_at_center", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[0],
                     style_order=pf_factors[0], data=motion_df, ci=ci_value, ax=ax13)

        sns.lineplot(x="timepoint", y="r_edge", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[0],
                     style_order=pf_factors[0], data=motion_df, ci=ci_value, ax=ax14)
        sns.lineplot(x="timepoint", y="r_dist", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[0],
                     style_order=pf_factors[0], data=motion_df, ci=ci_value, ax=ax15)
        sns.lineplot(x="timepoint", y="movelength_ratio_at_edge", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[0],
                     style_order=pf_factors[0], data=motion_df, ci=ci_value, ax=ax16)

        ax2.set_title(f'Locomotion', fontsize=14, fontweight='bold', color='#30302f', loc='center')
        ax1.set(ylabel="Total move length(mm)")
        ax2.set(ylabel="Move length at arena edge(mm)")
        ax3.set(ylabel="Move length at arena centre(mm)")

        ax5.set_title(f'Velocity', fontsize=14, fontweight='bold', color='#30302f', loc='center')

        ax4.set(ylabel="Average velocity(mm/s)")
        ax5.set(ylabel="Average velocity at arena edge(mm/s)")
        ax6.set(ylabel="Average velocity at arena centre(mm/s)")

        ax8.set(ylabel="Maximum velocity(mm/s)")
        ax9.set(ylabel="Maximum velocity at arena edge(mm/s)")
        ax10.set(ylabel="Maximum velocity at arena centre(mm/s)")

        ax12.set_title(f'Move time', fontsize=14, fontweight='bold', color='#30302f', loc='center')
        ax11.set(ylabel="Move time proportion(%)")
        ax12.set(ylabel="Move time proportion at arena edge(%)")
        ax13.set(ylabel="Move time proportion at arena centre(%)")

        ax15.set_title(f'Spatial preference', fontsize=14, fontweight='bold', color='#30302f', loc='center')
        ax14.set(ylabel="Time proportion spent at edge(%)")
        ax15.set(ylabel="Average distance from the arena center/Radius(%)")
        ax16.set(ylabel="Move length ratio at edge(%)")

        plt.tight_layout()
        pdf.savefig()
        plt.close()
        # page 2
        sns.set()
        plt.figure(figsize=(40, 26))
        # plt.rcParams['figure.constrained_layout.use'] = True
        ax1 = plt.subplot2grid((4, 12), (0, 0), colspan=4, rowspan=1)
        ax2 = plt.subplot2grid((4, 12), (0, 4), colspan=4, rowspan=1)
        ax3 = plt.subplot2grid((4, 12), (0, 8), colspan=4, rowspan=1)
        ax4 = plt.subplot2grid((4, 12), (1, 0), colspan=4, rowspan=1)
        ax5 = plt.subplot2grid((4, 12), (1, 4), colspan=4, rowspan=1)
        ax6 = plt.subplot2grid((4, 12), (1, 8), colspan=4, rowspan=1)
        ax7 = plt.subplot2grid((4, 12), (2, 0), colspan=4, rowspan=1)
        ax8 = plt.subplot2grid((4, 12), (2, 4), colspan=4, rowspan=1)
        ax9 = plt.subplot2grid((4, 12), (2, 8), colspan=4, rowspan=1)
        ax10 = plt.subplot2grid((4, 12), (3, 0), colspan=4, rowspan=1)
        ax11 = plt.subplot2grid((4, 12), (3, 4), colspan=4, rowspan=1)
        ax12 = plt.subplot2grid((4, 12), (3, 8), colspan=4, rowspan=1)
        motion_df['search_area_time'] = 100 * motion_df['search_area_time']
        sns.lineplot(x="timepoint", y="tracks_num", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[0],
                     style_order=pf_factors[0], data=motion_df, ci=ci_value, ax=ax1)
        sns.lineplot(x="timepoint", y="tracks_duration", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[0],
                     style_order=pf_factors[0], data=motion_df, ci=ci_value, ax=ax2)
        sns.lineplot(x="timepoint", y="tracks_length", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[0],
                     style_order=pf_factors[0], data=motion_df, ci=ci_value, ax=ax3)

        sns.lineplot(x="timepoint", y="stop_duration", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[0],
                     style_order=pf_factors[0], data=motion_df, ci=ci_value, ax=ax4)
        sns.lineplot(x="timepoint", y="long_stop_num", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[0],
                     style_order=pf_factors[0], data=motion_df, ci=ci_value, ax=ax5)
        sns.lineplot(x="timepoint", y="long_stop_duration", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[0],
                     style_order=pf_factors[0], data=motion_df, ci=ci_value, ax=ax6)

        sns.lineplot(x="timepoint", y="search_area_time", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[0],
                     style_order=pf_factors[0], data=motion_df, ci=ci_value, ax=ax7)
        sns.lineplot(x="timepoint", y="search_area", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[0],
                     style_order=pf_factors[0], data=motion_df, ci=ci_value, ax=ax8)
        sns.lineplot(x="timepoint", y="search_area_unit_move", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[0],
                     style_order=pf_factors[0], data=motion_df, ci=ci_value, ax=ax9)

        sns.lineplot(x="timepoint", y="track_straightness", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[0],
                     style_order=pf_factors[0], data=motion_df, ci=ci_value, ax=ax10)
        sns.lineplot(x="timepoint", y="track_straightness_at_edge", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[0],
                     style_order=pf_factors[0], data=motion_df, ci=ci_value, ax=ax11)
        sns.lineplot(x="timepoint", y="track_straightness_non_edge", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[0],
                     style_order=pf_factors[0], data=motion_df, ci=ci_value, ax=ax12)

        ax2.set_title(f'Tracks', fontsize=14, fontweight='bold', color='#30302f', loc='center')
        ax1.set(ylabel="Tracks number")
        ax2.set(ylabel="Average track duration(s)")
        ax3.set(ylabel="Average track length(mm)")

        ax5.set_title(f'Inactivity episodes', fontsize=14, fontweight='bold', color='#30302f', loc='center')
        ax4.set(ylabel="Average inactivity duration(s)")
        ax5.set(ylabel="Long inactivity episodes number")
        ax6.set(ylabel="Average long inactivity episodes duration(s)")

        ax8.set_title(f'Arena exploration', fontsize=14, fontweight='bold', color='#30302f', loc='center')
        ax7.set(ylabel="Time used to search area/Recorded time(%)")
        ax8.set(ylabel="Arena explored proportion at a given time")
        ax9.set(ylabel="Arena explored proportion/Travel length(/mm)")

        ax11.set_title(f'Track straightness', fontsize=14, fontweight='bold', color='#30302f', loc='center')
        ax10.set(ylabel="Average track straightness")
        ax11.set(ylabel="Track straightness at arena edge")
        ax12.set(ylabel="Track straightness at arena centre")

        # if plot_with_logged_yscale=='1':
        #     ax1.set(yscale="log")
        #     ax2.set(yscale="log")
        #     ax3.set(yscale="log")
        #     ax4.set(yscale="log")
        #     ax5.set(yscale="log")
        #     ax6.set(yscale="log")
        #     ax7.set(yscale="log")
        #     ax8.set(yscale="log")
        #     ax9.set(yscale="log")
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        # page 3
        sns.set()
        # plt.figure()
        fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(40, 26))
        sns.lineplot(x="timepoint", y="angular_velocity", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[0],
                     style_order=pf_factors[0], data=motion_df, ci=ci_value, ax=axes[0, 0])
        sns.lineplot(x="timepoint", y="angular_velocity_at_edge", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[0],
                     style_order=pf_factors[0], data=motion_df, ci=ci_value, ax=axes[0, 1])
        sns.lineplot(x="timepoint", y="angular_velocity_non_edge", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[0],
                     style_order=pf_factors[0], data=motion_df, ci=ci_value, ax=axes[0, 2])

        sns.lineplot(x="timepoint", y="max_angular_velocity", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[0],
                     style_order=pf_factors[0], data=motion_df, ci=ci_value, ax=axes[1, 0])
        sns.lineplot(x="timepoint", y="max_angular_velocity_at_edge", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[0],
                     style_order=pf_factors[0], data=motion_df, ci=ci_value, ax=axes[1, 1])
        sns.lineplot(x="timepoint", y="max_angular_velocity_non_edge", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[0],
                     style_order=pf_factors[0], data=motion_df, ci=ci_value, ax=axes[1, 2])

        sns.lineplot(x="timepoint", y="meander", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[0],
                     style_order=pf_factors[0], data=motion_df, ci=ci_value, ax=axes[2, 0])
        sns.lineplot(x="timepoint", y="meander_at_edge", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[0],
                     style_order=pf_factors[0], data=motion_df, ci=ci_value, ax=axes[2, 1])
        sns.lineplot(x="timepoint", y="meander_non_edge", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[0],
                     style_order=pf_factors[0], data=motion_df, ci=ci_value, ax=axes[2, 2])

        sns.lineplot(x="timepoint", y="max_meander", hue=pf_list[0], hue_order=pf_factors[0], style=pf_list[0],
                     style_order=pf_factors[0], data=motion_df, ci=ci_value, ax=axes[3, 0])
        sns.lineplot(x="timepoint", y="max_meander_at_edge", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[0],
                     style_order=pf_factors[0], data=motion_df, ci=ci_value, ax=axes[3, 1])
        sns.lineplot(x="timepoint", y="max_meander_non_edge", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[0],
                     style_order=pf_factors[0], data=motion_df, ci=ci_value, ax=axes[3, 2])

        axes[0, 1].set_title(f'Average angular', fontsize=14, fontweight='bold', color='#30302f', loc='center')
        axes[0, 0].set(ylabel="Angular velocity(rad/s)")
        axes[0, 1].set(ylabel="Angular velocity at arena edge(rad/s)")
        axes[0, 2].set(ylabel="Angular velocity at arena centre(rad/s)")

        axes[1, 1].set_title(f'Maximum angular', fontsize=14, fontweight='bold', color='#30302f', loc='center')
        axes[1, 0].set(ylabel="Maximum angular velocity(rad/s)")
        axes[1, 1].set(ylabel="Maximum angular velocity at arena edge(rad/s)")
        axes[1, 2].set(ylabel="Maximum angular velocity at arena centre(rad/s)")

        axes[2, 1].set_title(f'Meander', fontsize=14, fontweight='bold', color='#30302f', loc='center')
        axes[2, 0].set(ylabel="Meander(rad/mm)")
        axes[2, 1].set(ylabel="Meander at arena edge(rad/mm)")
        axes[2, 2].set(ylabel="Meander at arena centre(rad/mm)")

        axes[3, 1].set_title(f'Maximum meander', fontsize=14, fontweight='bold', color='#30302f', loc='center')
        axes[3, 0].set(ylabel="Maximum meander(rad/mm)")
        axes[3, 1].set(ylabel="Maximum meander velocity at arena edge(rad/mm)")
        axes[3, 2].set(ylabel="Maximum meander velocity at arena centre(rad/mm)")

        if plot_with_logged_yscale == '1':
            axes[2, 0].set(yscale="log")
            axes[2, 1].set(yscale="log")
            axes[2, 2].set(yscale="log")
            axes[3, 0].set(yscale="log")
            axes[3, 1].set(yscale="log")
            axes[3, 2].set(yscale="log")

        #     for i in range(4):
        #         for j in range(3):
        #             axes[i,j].set(yscale="log")
        plt.tight_layout()
        pdf.savefig()
        plt.close()
    with PdfPages(interaction_image) as pdf:
        # page 4
        sns.set()
        # plt.figure()
        fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(40, 20))
        sns.lineplot(x="timepoint", y="distance_space", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[0],
                     style_order=pf_factors[0], data=interaction_df, ci=ci_value, ax=axes[0, 0])

        sns.lineplot(x="timepoint", y="distance_space_at_edge", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[0],
                     style_order=pf_factors[0], data=interaction_df, ci=ci_value, ax=axes[1, 0])
        sns.lineplot(x="timepoint", y="distance_space_at_center", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[0],
                     style_order=pf_factors[0], data=interaction_df, ci=ci_value, ax=axes[1, 1])
        sns.lineplot(x="timepoint", y="distance_space_at_move", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[0],
                     style_order=pf_factors[0], data=interaction_df, ci=ci_value, ax=axes[1, 2])
        sns.lineplot(x="timepoint", y="distance_space_at_stop", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[0],
                     style_order=pf_factors[0], data=interaction_df, ci=ci_value, ax=axes[1, 3])

        sns.lineplot(x="timepoint", y="SSI", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[0],
                     style_order=pf_factors[0], data=interaction_df, ci=ci_value, ax=axes[2, 0])

        sns.lineplot(x="timepoint", y="SSI_at_edge", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[0],
                     style_order=pf_factors[0], data=interaction_df, ci=ci_value, ax=axes[3, 0])
        sns.lineplot(x="timepoint", y="SSI_at_center", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[0],
                     style_order=pf_factors[0], data=interaction_df, ci=ci_value, ax=axes[3, 1])
        sns.lineplot(x="timepoint", y="SSI_at_move", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[0],
                     style_order=pf_factors[0], data=interaction_df, ci=ci_value, ax=axes[3, 2])
        sns.lineplot(x="timepoint", y="SSI_at_stop", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[0],
                     style_order=pf_factors[0], data=interaction_df, ci=ci_value, ax=axes[3, 3])

        axes[0, 1].axis('off')
        axes[0, 2].axis('off')
        axes[0, 3].axis('off')
        axes[2, 1].axis('off')
        axes[2, 2].axis('off')
        axes[2, 3].axis('off')
        axes[0, 1].set_title(f'Social distance space', fontsize=14, fontweight='bold', color='#30302f', loc='right')

        axes[0, 0].set(ylabel="Distance space(mm)")

        axes[1, 0].set(ylabel="Distance space at arena edge(mm)")
        axes[1, 1].set(ylabel="Distance space at arena centre(mm)")
        axes[1, 2].set(ylabel="Distance space at activity episodes(mm)")
        axes[1, 3].set(ylabel="Distance space at inactivity episodes(mm)")

        axes[2, 1].set_title(f'Social space index(SSI)', fontsize=14, fontweight='bold', color='#30302f', loc='right')

        axes[3, 0].set(ylabel="SSI at arena edge")
        axes[3, 1].set(ylabel="SSI at arena centre")
        axes[3, 2].set(ylabel="SSI at activity episodes")
        axes[3, 3].set(ylabel="SSI at inactivity episodes")

        # if plot_with_logged_yscale=='1':
        #     for i in range(4):
        #         for j in range(3):
        #             axes[i,j].set(yscale="log")
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        # page 5
        sns.set()
        fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(40, 20))
        interaction_df['interaction'] = 100 * interaction_df['interaction']
        interaction_df['interaction_at_edge_proportion'] = 100 * interaction_df['interaction_at_edge_proportion']
        interaction_df['interaction_at_center_proportion'] = 100 * interaction_df['interaction_at_center_proportion']
        interaction_df['interaction_at_edge'] = 100 * interaction_df['interaction_at_edge']
        interaction_df['interaction_at_center'] = 100 * interaction_df['interaction_at_center']

        interaction_df['interaction_at_move'] = 100 * interaction_df['interaction_at_move']
        interaction_df['interaction_at_stop'] = 100 * interaction_df['interaction_at_stop']
        interaction_df['interaction_at_move_proportion'] = 100 * interaction_df['interaction_at_move_proportion']
        interaction_df['interaction_at_stop_proportion'] = 100 * interaction_df['interaction_at_stop_proportion']

        interaction_df['interaction_at_long_stop'] = 100 * interaction_df['interaction_at_long_stop']
        interaction_df['interaction_at_long_stop_proportion'] = 100 * interaction_df[
            'interaction_at_long_stop_proportion']

        sns.lineplot(x="timepoint", y="interaction", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[0],
                     style_order=pf_factors[0], data=interaction_df, ci=ci_value, ax=axes[0, 0])
        sns.lineplot(x="timepoint", y="interaction_counts", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[0],
                     style_order=pf_factors[0], data=interaction_df, ci=ci_value, ax=axes[0, 1])
        sns.lineplot(x="timepoint", y="interaction_duration", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[0],
                     style_order=pf_factors[0], data=interaction_df, ci=ci_value, ax=axes[0, 2])
        sns.lineplot(x="timepoint", y="interaction_members", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[0],
                     style_order=pf_factors[0], data=interaction_df, ci=ci_value, ax=axes[0, 3])

        sns.lineplot(x="timepoint", y="interaction_at_edge", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[0],
                     style_order=pf_factors[0], data=interaction_df, ci=ci_value, ax=axes[1, 0])
        sns.lineplot(x="timepoint", y="interaction_at_center", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[0],
                     style_order=pf_factors[0], data=interaction_df, ci=ci_value, ax=axes[1, 1])
        sns.lineplot(x="timepoint", y="interaction_at_edge_proportion", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[0],
                     style_order=pf_factors[0], data=interaction_df, ci=ci_value, ax=axes[1, 2])
        sns.lineplot(x="timepoint", y="interaction_at_center_proportion", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[0],
                     style_order=pf_factors[0], data=interaction_df, ci=ci_value, ax=axes[1, 3])

        sns.lineplot(x="timepoint", y="interaction_at_move", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[0],
                     style_order=pf_factors[0], data=interaction_df, ci=ci_value, ax=axes[2, 0])
        sns.lineplot(x="timepoint", y="interaction_at_stop", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[0],
                     style_order=pf_factors[0], data=interaction_df, ci=ci_value, ax=axes[2, 1])
        sns.lineplot(x="timepoint", y="interaction_at_move_proportion", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[0],
                     style_order=pf_factors[0], data=interaction_df, ci=ci_value, ax=axes[2, 2])
        sns.lineplot(x="timepoint", y="interaction_at_stop_proportion", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[0],
                     style_order=pf_factors[0], data=interaction_df, ci=ci_value, ax=axes[2, 3])

        sns.lineplot(x="timepoint", y="interaction_at_long_stop", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[0],
                     style_order=pf_factors[0], data=interaction_df, ci=ci_value, ax=axes[3, 0])
        sns.lineplot(x="timepoint", y="interaction_at_long_stop_proportion", hue=pf_list[0], hue_order=pf_factors[0],
                     style=pf_list[0],
                     style_order=pf_factors[0], data=interaction_df, ci=ci_value, ax=axes[3, 1])

        # axes[3,1].axis('off')
        axes[3, 2].axis('off')
        axes[3, 3].axis('off')
        axes[0, 1].set_title(f'Interaction', fontsize=14, fontweight='bold', color='#30302f', loc='right')

        axes[0, 0].set(ylabel="Total interaction duration(%)")
        axes[0, 1].set(ylabel="Interaction episode counts")
        axes[0, 2].set(ylabel="Interaction episode duration(s)")
        axes[0, 3].set(ylabel="Average number of crowed drosophila at interaction")

        axes[1, 0].set(ylabel="Total interaction duration at arena edge(%)")
        axes[1, 1].set(ylabel="Total interaction duration at arena centre(%)")
        axes[1, 2].set(ylabel="The time proportion for interaction at edge(%)")
        axes[1, 3].set(ylabel="The time proportion for interaction  at centre(%)")

        axes[2, 0].set(ylabel="Total interaction duration at activity episodes(%)")
        axes[2, 1].set(ylabel="Total interaction duration at inactivity episodes(%)")
        axes[2, 2].set(ylabel="The time proportion for interaction at activity episodes(%)")
        axes[2, 3].set(ylabel="The time proportion for interaction at inactivity episodes(%)")

        axes[3, 0].set(ylabel="Total interaction duration at long-stop(%)")
        axes[3, 1].set(ylabel="The time proportion for interaction at long-stop(%)")

        plt.tight_layout()
        pdf.savefig()
        plt.close()
        # page 6
        sns.set()
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(40, 14))

        if plot_by_video == '1':
            interaction_df['unconnected_social_network_proportion'] = 100 * interaction_df[
                'unconnected_social_network_proportion']
            sns.lineplot(x="timepoint", y="acquaintances", hue=pf_list[0],
                         hue_order=pf_factors[0],
                         style=pf_list[0],
                         style_order=pf_factors[0], data=interaction_df, ci=ci_value, ax=axes[0, 0])
            sns.lineplot(x="timepoint", y="degree", hue=pf_list[0],
                         hue_order=pf_factors[0],
                         style=pf_list[0],
                         style_order=pf_factors[0], data=interaction_df, ci=ci_value, ax=axes[0, 1])
            sns.lineplot(x="timepoint", y="diameter", hue=pf_list[0],
                         hue_order=pf_factors[0],
                         style=pf_list[0],
                         style_order=pf_factors[0], data=interaction_df, ci=ci_value, ax=axes[0, 2])
            sns.lineplot(x="timepoint", y="unconnected_social_network_proportion", hue=pf_list[0],
                         hue_order=pf_factors[0],
                         style=pf_list[0],
                         style_order=pf_factors[0], data=interaction_df, ci=ci_value, ax=axes[0, 3])

            sns.lineplot(x="timepoint", y="degree_assortativity_coefficient", hue=pf_list[0],
                         hue_order=pf_factors[0],
                         style=pf_list[0],
                         style_order=pf_factors[0], data=interaction_df, ci=ci_value, ax=axes[1, 0])
            sns.lineplot(x="timepoint", y="clustering_coefficient", hue=pf_list[0],
                         hue_order=pf_factors[0],
                         style=pf_list[0],
                         style_order=pf_factors[0], data=interaction_df, ci=ci_value, ax=axes[1, 1])
            sns.lineplot(x="timepoint", y="betweenness_centrality", hue=pf_list[0],
                         hue_order=pf_factors[0],
                         style=pf_list[0],
                         style_order=pf_factors[0], data=interaction_df, ci=ci_value, ax=axes[1, 2])
            sns.lineplot(x="timepoint", y="global_efficiency", hue=pf_list[0],
                         hue_order=pf_factors[0],
                         style=pf_list[0],
                         style_order=pf_factors[0], data=interaction_df, ci=ci_value, ax=axes[1, 3])

        else:
            sns.lineplot(x="timepoint", y="acquaintances", hue=pf_list[0],
                         hue_order=pf_factors[0],
                         style=pf_list[0],
                         style_order=pf_factors[0], data=interaction_df, ci=ci_value, ax=axes[0, 0])
            sns.lineplot(x="timepoint", y="degree", hue=pf_list[0],
                         hue_order=pf_factors[0],
                         style=pf_list[0],
                         style_order=pf_factors[0], data=interaction_df, ci=ci_value, ax=axes[0, 1])
            sns.lineplot(x="timepoint", y="dominating", hue=pf_list[0],
                         hue_order=pf_factors[0],
                         style=pf_list[0],
                         style_order=pf_factors[0], data=interaction_df, ci=ci_value, ax=axes[0, 2])

            sns.lineplot(x="timepoint", y="closeness_centrality", hue=pf_list[0],
                         hue_order=pf_factors[0],
                         style=pf_list[0],
                         style_order=pf_factors[0], data=interaction_df, ci=ci_value, ax=axes[1, 0])
            sns.lineplot(x="timepoint", y="clustering_coefficient", hue=pf_list[0],
                         hue_order=pf_factors[0],
                         style=pf_list[0],
                         style_order=pf_factors[0], data=interaction_df, ci=ci_value, ax=axes[1, 1])
            sns.lineplot(x="timepoint", y="betweenness_centrality", hue=pf_list[0],
                         hue_order=pf_factors[0],
                         style=pf_list[0],
                         style_order=pf_factors[0], data=interaction_df, ci=ci_value, ax=axes[1, 2])
            sns.lineplot(x="timepoint", y="eccentricity", hue=pf_list[0],
                         hue_order=pf_factors[0],
                         style=pf_list[0],
                         style_order=pf_factors[0], data=interaction_df, ci=ci_value, ax=axes[1, 3])

        axes[0, 1].set_title(f'Social networks', fontsize=14, fontweight='bold', color='#30302f', loc='right')
        if plot_by_video == '1':
            axes[0, 0].set(ylabel="Acquaintances")
            axes[0, 1].set(ylabel="Network degree")
            axes[0, 2].set(ylabel="Network diameter")
            axes[0, 3].set(ylabel="Unconnected social network proportion(%)")

            axes[1, 0].set(ylabel="Degree assortativity coefficient")
            axes[1, 1].set(ylabel="Clustering coefficient")
            axes[1, 2].set(ylabel="Betweenness centrality")
            axes[1, 3].set(ylabel="Global efficiency")
        else:
            axes[0, 0].set(ylabel="Acquaintances")
            axes[0, 1].set(ylabel="Network degree")
            axes[0, 2].set(ylabel="Dominating")
            axes[0, 3].axis('off')
            axes[1, 0].set(ylabel="Closeness centrality")
            axes[1, 1].set(ylabel="Clustering coefficient")
            axes[1, 2].set(ylabel="Betweenness centrality")
            axes[1, 3].set(ylabel="Eccentricity")
        plt.tight_layout()
        pdf.savefig()
        plt.close()