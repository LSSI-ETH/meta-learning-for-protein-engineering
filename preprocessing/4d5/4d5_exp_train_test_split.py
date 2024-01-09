#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: minotm
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from preprocessing_utils import *

data_type = '4d5_exp'

def class_balance_3class_4d5(df, use_LD = False, islist = False):
    
    if islist == True:
        df = combine_df_list_and_shuffle(df, keep=False)

    elif islist == False:
        df = df.copy()
        high = df[df['AgClass'] == 2]
        low = df[df['AgClass'] == 1]
        negatives = df[df['AgClass'] == 0]
        df = combine_df_list_and_shuffle([high, low, negatives], keep=False)
        
    high = df[df['AgClass'] == 2]
    low = df[df['AgClass'] == 1]
    negatives = df[df['AgClass'] == 0]
    
    truncate_value = min([len(ls) for ls in [high, low, negatives]])
    
    if use_LD == False:
        high = high.sample(n = int(np.round(truncate_value)))
        low = low.sample(n = int(np.round(truncate_value)))
        negatives = negatives.sample(n = int(np.round(truncate_value)))
        
    elif use_LD == True:
        truncate_value = min([len(ls) for ls in [high, low, negatives]])
        if len(high) > truncate_value:
            discarded_seqs, high, high_train, y_discard = train_test_split(high, high['AgClass'], test_size = truncate_value,
                                                                      random_state = 1, shuffle = True, stratify = high['LD'])
        if len(low) > truncate_value:
            low = low[low['LD'] != 1] #remove LD 1 for stratification on edit distance
            discarded_seqs, low, low_train, y_discard = train_test_split(low, low['AgClass'], test_size = truncate_value,
                                                                      random_state = 1, shuffle = True, stratify = low['LD'])
        if len(negatives) > truncate_value:
            negatives = negatives[negatives['LD'] != 0] #remove LD 0 for stratification on edit distance
            discarded_seqs, negatives, negatives_train, y_discard = train_test_split(negatives, negatives['AgClass'], test_size = truncate_value,
                                                                      random_state = 1, shuffle = True, stratify = negatives['LD'])   
    return combine_df_list_and_shuffle([high, low, negatives], keep=False)


print('Now splitting 4d5 expeirmental noise data into training & test sets...')
path_4d5 = '../../data/4d5/'

#load clean data
high_data = pd.read_csv(path_4d5 + '4d5_high_4th_sort.csv')
low_data = pd.read_csv(path_4d5 + '4d5_low_4th_sort.csv')
neg_data_full = pd.read_csv(path_4d5 + '4d5_neg_4th_sort.csv')
#remove sequences with proline at position 7
neg_data = neg_data_full[~neg_data_full['AASeq'].str.contains(r'^.{7}[P]')].copy()


#load initial sort data, suffix d corresponds to dirty (noisy) data
high_data_d = pd.read_csv(path_4d5 + '4d5_high_initial_sort.csv')
low_data_d = pd.read_csv(path_4d5 + '4d5_low_initial_sort.csv')
neg_data_full_d = pd.read_csv(path_4d5 + '4d5_neg_initial_sort.csv')
#remove sequences with proline at position 7
neg_data_d = neg_data_full_d[~neg_data_full_d['AASeq'].str.contains(r'^.{7}[P]')].copy()

high_data['AgClass'], high_data_d['AgClass'] = 2,2
low_data['AgClass'], low_data_d['AgClass']  = 1, 1
neg_data['AgClass'], neg_data_d['AgClass'] = 0, 0

high_data = high_data.sample(frac = 1, random_state = 1)
low_data = low_data.sample(frac = 1, random_state = 1)
neg_data = neg_data.sample(frac = 1, random_state = 1)

high_data_d = high_data_d.sample(frac = 1, random_state = 1)
low_data_d = low_data_d.sample(frac = 1, random_state = 1)
neg_data_d = neg_data_d.sample(frac = 1, random_state = 1)

neg_data['df'], low_data['df'], high_data['df'] = 'clean', 'clean', 'clean'
neg_data_d['df'], low_data_d['df'], high_data_d['df'] = 'dirty', 'dirty', 'dirty'

data_frame =  combine_df_list_and_shuffle([high_data,low_data, neg_data, high_data_d, low_data_d, neg_data_d],keep = False)
data_frame_d = data_frame[data_frame['df'] == 'dirty']
data_frame = data_frame[data_frame['df'] == 'clean']

data_frame = class_balance_3class_4d5(data_frame, use_LD = False, islist = False)
data_frame_d = class_balance_3class_4d5(data_frame_d, use_LD = False, islist = False)

data_frame = pd.concat([data_frame, data_frame_d], ignore_index = True)

data_frame, _ = consensus_seq_and_LD_to_df(data_frame = data_frame, seq_col_str = 'AASeq', return_df = True)

data_frame_d = data_frame[data_frame['df'] == 'dirty']
data_frame = data_frame[data_frame['df'] == 'clean']

#split train & test set by edit distance, class balance, & save to csv
for ed_thresh in [4,5,6,7,8]:
    
    train_test_LD_split_threshold = ed_thresh
    train_df = data_frame[data_frame['LD'] <= train_test_LD_split_threshold]
    test_df = data_frame[data_frame['LD'] > train_test_LD_split_threshold]
    
    train_df = train_df[(train_df['LD'] != 2) & (train_df['LD'] != 1)]
    train_df = class_balance_3class_4d5(train_df, use_LD = True, islist = False)
    test_df = class_balance_3class_4d5(test_df, use_LD = True, islist = False)
    
    train_df_d = data_frame_d[data_frame_d['LD'] <= train_test_LD_split_threshold]
    test_df_d = data_frame_d[data_frame_d['LD'] > train_test_LD_split_threshold]
    
    train_df_d = class_balance_3class_4d5(train_df_d, use_LD = True, islist = False)
    test_df_d = class_balance_3class_4d5(test_df_d, use_LD = True, islist = False)
    
    #separate into classes for data set splitting
    train_high = train_df[(train_df['AgClass'] == 2)]
    train_low = train_df[(train_df['AgClass'] == 1)] 
    train_neg = train_df[train_df['AgClass'] == 0]
    
    #allocate 25% training sequences for use in validation and meta sets
    train_high, val_meta_high, _, _ = train_test_split(train_high, train_high['AgClass'], test_size = 0.25,
                                                              random_state = 1, shuffle = True, stratify = train_high['LD'])
    
    val_high, meta_high, _, _ = train_test_split(val_meta_high, val_meta_high['AgClass'], test_size = 0.20,
                                                              random_state = 1, shuffle = True, stratify = val_meta_high['LD'])
    
    train_low, val_meta_low, _, _ = train_test_split(train_low, train_low['AgClass'], test_size = 0.25,
                                                              random_state = 1, shuffle = True, stratify = train_low['LD'])
    
    val_low, meta_low, _, _ = train_test_split(val_meta_low, val_meta_low['AgClass'], test_size = 0.20,
                                                              random_state = 1, shuffle = True, stratify = val_meta_low['LD'])
    
    
    train_neg, val_meta_neg, _, _ = train_test_split(train_neg, train_neg['AgClass'], test_size = 0.25,
                                                              random_state = 1, shuffle = True, stratify = train_neg['LD'])
    
    val_neg, meta_neg, _, _ = train_test_split(val_meta_neg, val_meta_neg['AgClass'], test_size = 0.20,
                                                              random_state = 1, shuffle = True, stratify = val_meta_neg['LD'])
    
    train_df = pd.concat([train_high, train_low, train_neg], ignore_index = True).sample(frac = 1, random_state = 1)
    val_df = pd.concat([val_high, val_low, val_neg], ignore_index = True).sample(frac = 1, random_state = 1)
    meta_df = pd.concat([meta_high, meta_low, meta_neg], ignore_index = True).sample(frac = 1, random_state = 1)
    
    train_df_d = class_balance_3class_4d5(train_df_d, use_LD = False, islist = False)
    #val_df = class_balance_3class_4d5(val_df, use_LD = False, islist = False)
    
    
    train_df.to_csv(path_4d5 + f'{data_type}_train_clean_ed_{ed_thresh}.csv', index=False, header=True)
    train_df_d.to_csv(path_4d5 + f'{data_type}_ed_{ed_thresh}_train.csv', index=False, header=True)
    val_df.to_csv(path_4d5 + f'{data_type}_ed_{ed_thresh}_val.csv', index=False, header=True)
    test_df.to_csv(path_4d5 + f'{data_type}_ed_{ed_thresh}_test.csv', index=False, header=True)
    meta_df.to_csv(path_4d5 + f'{data_type}_ed_{ed_thresh}_meta.csv', index=False, header=True)
    
    print('Data splitting completed')