#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split

data_type = '4d5_exp'

#Function to drop duplicates from two dataframes
def drop_test_seqs(train_df, test_df, seq_name):
    train_df = train_df.copy()
    train_df['df'] = 'train'
    test_df = test_df.copy()
    test_df['df'] = 'test'
    frames = [train_df.copy(),test_df.copy()]
    common_cols = list(set.intersection(*(set(df.columns) for df in frames)))
    concat_df = pd.concat([df[common_cols] for df in frames], ignore_index=True)
    concat_df = concat_df.drop_duplicates(subset=[seq_name],keep=False)
    out_df = concat_df[concat_df['df'] == 'train']
    out_df.drop(columns = ['df'])
    return out_df

def batch_meta_set(meta_df_full):
    '''
    
    Parameters
    ----------
    train_truncated : pandas dataframe
        dataframe contianing training data with corrupted labels.
    train_clean : pandas dataframe
        dataframe containing training data with trusted labels.
    truncate_factor : float
        factor by which the training set is being truncated.
    dirty_meta : BOOL
            Determines whether meta set should be batched from training set containining trusted or corrupted labels.

    Returns
    -------
    None. Saves different meta sets to csv file
    '''
    
    meta_size_dict = {32:0, 96:1, 288:2, 864:3}
    for meta_size in [32, 96, 288, 864]:
        #batch meta set from verified double classified seqs
        meta_seqs_high = meta_df_full[meta_df_full['AgClass'] == 2]
        meta_seqs_low = meta_df_full[meta_df_full['AgClass'] == 1]
        meta_seqs_neg = meta_df_full[meta_df_full['AgClass'] == 0]
        
        meta_df = pd.DataFrame()
        
        meta_df_list = [meta_seqs_high, meta_seqs_low, meta_seqs_neg]
        
        for i in range(len(meta_df_list)):
            
            meta_partition =  int(meta_size/  3)
                
            train_tmp, meta_tmp, y_train, y_meta = train_test_split(meta_df_list[i], meta_df_list[i]['AgClass'], test_size =meta_partition,
                                                                      random_state = 1, shuffle = True, stratify = meta_df_list[i]['LD'])
            
            meta_df = pd.concat([meta_df, meta_tmp], ignore_index = True)
            
        meta_df = meta_df.sample(frac = 1, random_state = 1)
        meta_out_str = data_path  + f'{data_type}_meta_set_{str(meta_size_dict[meta_size])}.csv'
        meta_df.to_csv(meta_out_str, index=False, header = True)
         
        
#================================== Load Data  ===================================
data_path = '../../data/4d5/'
test_full = pd.read_csv(data_path + f'{data_type}_test.csv')
train_full = pd.read_csv(data_path + f'{data_type}_train.csv')
val_full = pd.read_csv(data_path + f'{data_type}_val.csv')
meta_full = pd.read_csv(data_path + f'{data_type}_meta.csv')

train_full = train_full[train_full['LD'] != 1]

#========================= Truncate Dirty Training Set ===============================
train_full = drop_test_seqs(train_full, test_full, 'AASeq')
train_full = drop_test_seqs(train_full, val_full, 'AASeq')

truncate_factor_list = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]

for truncate_factor in truncate_factor_list:

    train = train_full.copy()
    if truncate_factor != 1.0:
        
        train_high = train[train['AgClass'] == 2]
        train_low = train[train['AgClass'] == 1]
        train_neg = train[train['AgClass'] == 0]
        
        train_truncated = pd.DataFrame()
       
        train_df_list = [train_high, train_low, train_neg]
       
        for i in range(len(train_df_list)):
                
            train_tmp, meta_tmp, y_train, y_meta = train_test_split(train_df_list[i], train_df_list[i]['AgClass'], test_size =1 - truncate_factor,
                                                                      random_state = 1, shuffle = True, stratify = train_df_list[i]['LD'])
            
            train_truncated = pd.concat([train_truncated, train_tmp], ignore_index = True)

    elif truncate_factor == 1.0:
        train_truncated = train
    
    #train_truncated = train_truncated[train_truncated['LD'] != 2] #drop single LD 2 sequence for sklearn train test split by edit distance
    train_truncated = train_truncated.drop(columns = ['df'])
    out_str = data_path  + f'{data_type}_train_truncated_{str(truncate_factor)}.csv'
    train_truncated.to_csv(out_str, index=False, header = True) 

#========================= Batch Clean Meta Setss ===============================
batch_meta_set(meta_df_full = meta_full)