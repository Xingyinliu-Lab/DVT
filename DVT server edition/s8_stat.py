import pandas as pd
import numpy as np
import os
import sys
# fileplace = sys.argv[1]
# metadata_name = sys.argv[2]

fileplace =sys.argv[1]
predix=sys.argv[2]
metadata_name = 'metadata.csv'
fileplace_analysis = fileplace+'analysis'+predix+'/'
metadata = pd.read_csv(fileplace + metadata_name, header=0)
ind_montion_stat_df=pd.DataFrame()
ind_interaction_stat_df=pd.DataFrame()
avg_montion_stat_df=pd.DataFrame()
avg_interaction_stat_df=pd.DataFrame()

sliced_ind_montion_stat_df=pd.DataFrame()
sliced_ind_interaction_stat_df=pd.DataFrame()
sliced_avg_montion_stat_df=pd.DataFrame()
sliced_avg_interaction_stat_df=pd.DataFrame()



for meta_index in metadata.index:
    replicate=metadata.loc[meta_index, 'replicate']
    sex=metadata.loc[meta_index, 'sex']
    num=metadata.loc[meta_index, 'num']
    condition=metadata.loc[meta_index, 'condition']
    v_name = metadata.loc[meta_index, 'videoname']
    genotype = metadata.loc[meta_index, 'genotype']
    if os.path.exists(fileplace_analysis + v_name+'_ind_motion.csv'):
        ind=pd.read_csv(fileplace_analysis + v_name+'_ind_motion.csv')
        ind['num']=num
        ind['sex']=sex
        ind['replicate']=replicate
        ind['condition']=condition
        ind['genotype']=genotype
        if len(ind_montion_stat_df)>0:
            ind_montion_stat_df=pd.concat([ind, ind_montion_stat_df])
        if len(ind_montion_stat_df)==0:
            ind_montion_stat_df=ind
    if os.path.exists(fileplace_analysis + v_name + '_ind_interaction.csv'):
        interaction=pd.read_csv(fileplace_analysis + v_name + '_ind_interaction.csv')
        interaction['num']=num
        interaction['sex']=sex
        interaction['replicate']=replicate
        interaction['condition']=condition
        interaction['genotype']=genotype
        if len(ind_interaction_stat_df)>0:
            ind_interaction_stat_df=pd.concat([interaction, ind_interaction_stat_df])
        if len(ind_interaction_stat_df)==0:
            ind_interaction_stat_df=interaction
    if os.path.exists(fileplace_analysis + v_name + '_avg_interaction.csv'):
        interaction=pd.read_csv(fileplace_analysis + v_name + '_avg_interaction.csv')
        interaction['num']=num
        interaction['sex']=sex
        interaction['replicate']=replicate
        interaction['condition']=condition
        interaction['genotype']=genotype
        if len(avg_interaction_stat_df)>0:
            avg_interaction_stat_df=pd.concat([interaction, avg_interaction_stat_df])
        if len(avg_interaction_stat_df)==0:
            avg_interaction_stat_df=interaction
    if os.path.exists(fileplace_analysis + v_name + '_avg_motion.csv'):
        interaction=pd.read_csv(fileplace_analysis + v_name + '_avg_motion.csv')
        interaction['num']=num
        interaction['sex']=sex
        interaction['replicate']=replicate
        interaction['condition']=condition
        interaction['genotype']=genotype
        if len(avg_montion_stat_df)>0:
            avg_montion_stat_df=pd.concat([interaction, avg_montion_stat_df])
        if len(avg_montion_stat_df)==0:
            avg_montion_stat_df=interaction

    if os.path.exists(fileplace_analysis + v_name+'_ind_motion_sliced.csv'):
        ind=pd.read_csv(fileplace_analysis + v_name+'_ind_motion_sliced.csv')
        ind['num']=num
        ind['sex']=sex
        ind['replicate']=replicate
        ind['condition']=condition
        ind['genotype']=genotype
        if len(sliced_ind_montion_stat_df)>0:
            sliced_ind_montion_stat_df=pd.concat([ind, sliced_ind_montion_stat_df])
        if len(sliced_ind_montion_stat_df)==0:
            sliced_ind_montion_stat_df=ind
    if os.path.exists(fileplace_analysis + v_name + '_ind_interaction_sliced.csv'):
        interaction=pd.read_csv(fileplace_analysis + v_name + '_ind_interaction_sliced.csv')
        interaction['num']=num
        interaction['sex']=sex
        interaction['replicate']=replicate
        interaction['condition']=condition
        interaction['genotype']=genotype
        if len(sliced_ind_interaction_stat_df)>0:
            sliced_ind_interaction_stat_df=pd.concat([interaction, sliced_ind_interaction_stat_df])
        if len(sliced_ind_interaction_stat_df)==0:
            sliced_ind_interaction_stat_df=interaction
    if os.path.exists(fileplace_analysis + v_name + '_avg_interaction_sliced.csv'):
        interaction=pd.read_csv(fileplace_analysis + v_name + '_avg_interaction_sliced.csv')
        interaction['num']=num
        interaction['sex']=sex
        interaction['replicate']=replicate
        interaction['condition']=condition
        interaction['genotype']=genotype
        if len(sliced_avg_interaction_stat_df)>0:
            sliced_avg_interaction_stat_df=pd.concat([interaction, sliced_avg_interaction_stat_df])
        if len(sliced_avg_interaction_stat_df)==0:
            sliced_avg_interaction_stat_df=interaction
    if os.path.exists(fileplace_analysis + v_name + '_avg_motion_sliced.csv'):
        interaction=pd.read_csv(fileplace_analysis + v_name + '_avg_motion_sliced.csv')
        interaction['num']=num
        interaction['sex']=sex
        interaction['replicate']=replicate
        interaction['condition']=condition
        interaction['genotype']=genotype
        if len(sliced_avg_montion_stat_df)>0:
            sliced_avg_montion_stat_df=pd.concat([interaction, sliced_avg_montion_stat_df])
        if len(sliced_avg_montion_stat_df)==0:
            sliced_avg_montion_stat_df=interaction

ind_montion_stat_df.to_csv(fileplace_analysis + 'ind_motion.csv')
ind_interaction_stat_df.to_csv(fileplace_analysis + 'ind_interaction.csv')
avg_montion_stat_df.to_csv(fileplace_analysis + 'avg_motion.csv')
avg_interaction_stat_df.to_csv(fileplace_analysis + 'avg_interaction.csv')


sliced_ind_montion_stat_df.to_csv(fileplace_analysis + 'ind_motion_sliced.csv')
sliced_ind_interaction_stat_df.to_csv(fileplace_analysis + 'ind_interaction_sliced.csv')
sliced_avg_montion_stat_df.to_csv(fileplace_analysis + 'avg_motion_sliced.csv')
sliced_avg_interaction_stat_df.to_csv(fileplace_analysis + 'avg_interaction_sliced.csv')



