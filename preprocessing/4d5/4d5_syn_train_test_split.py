#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from preprocessing_utils import *

data_type = '4d5_syn'

def class_balance_3class_4d5(df, use_LD = False, islist = False):
    
    if islist == True:
        high = df[0].copy()
        low = df[1].copy()
        negatives = df[2].copy()
    elif islist == False:
        df = df.copy()
        high = df[df['AgClass'] == 2]
        low = df[df['AgClass'] == 1]
        negatives = df[df['AgClass'] == 0]
    
    truncate_value = min([len(ls) for ls in [high, low, negatives]])
    
    if use_LD == False:
        high = high[: int(np.round(truncate_value))] 
        low = low[: int(np.round(truncate_value))] 
        negatives = negatives[: int(np.round(truncate_value))] 
        
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

print('Now splitting clean 4d5 Data into training & test sets...')
path_4d5 = '../../data/4d5/'
high_data = pd.read_csv(path_4d5 + '4d5_high_4th_sort.csv')
low_data = pd.read_csv(path_4d5 + '4d5_low_4th_sort.csv')
neg_data_full = pd.read_csv(path_4d5 + '4d5_neg_4th_sort.csv')
#remove sequences with proline at position 7 as high frequencey of this residue over-simplifies learning
neg_data = neg_data_full[~neg_data_full['AASeq'].str.contains(r'^.{7}[P]')].copy()

high_data['AgClass'] = 2
low_data['AgClass'] = 1
neg_data['AgClass'] = 0

high_data = high_data.sample(frac = 1, random_state = 1)
low_data = low_data.sample(frac = 1, random_state = 1)
neg_data = neg_data.sample(frac = 1, random_state = 1)

data_frame = class_balance_3class_4d5([high_data, low_data, neg_data], use_LD = False, islist = True)

data_frame, _ = consensus_seq_and_LD_to_df(data_frame = data_frame, seq_col_str = 'AASeq', return_df = True)

#split train & test set by edit distance, class balance, & save to csv
train_test_LD_split_threshold = 6
train_df = data_frame[data_frame['LD'] <= train_test_LD_split_threshold]
test_df = data_frame[data_frame['LD'] > train_test_LD_split_threshold]

train_df = class_balance_3class_4d5(train_df, use_LD = True, islist = False)
test_df = class_balance_3class_4d5(test_df, use_LD = True, islist = False)

train_df = train_df[train_df['LD'] != 2]

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


train_df = class_balance_3class_4d5(train_df, use_LD = False, islist = False)
val_df = class_balance_3class_4d5(val_df, use_LD = False, islist = False)

train_df.to_csv(path_4d5 + f'{data_type}_train.csv', index=False, header=True)
val_df.to_csv(path_4d5 + f'{data_type}_val.csv', index=False, header=True)
test_df.to_csv(path_4d5 + f'{data_type}_test.csv', index=False, header=True)
meta_df.to_csv(path_4d5 + f'{data_type}_meta.csv', index=False, header=True)

print('Data splitting completed')
