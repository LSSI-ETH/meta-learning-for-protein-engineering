#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing_utils import *

data_type = '5a12_PUL_exp'

def combine_df_and_retain_duplicates(df_list):
    '''
    combines two dataframes, retains only duplicate amino acid sequences, & shuffles
    
    Parameters
    ----------

    data_frame : pandas.DataFrame
        dataframe containing all sequence & label data

    Returns
    -------
    data_frame : pandas.DataFrame
        combined, shuffled dataframe
    '''
    frames = df_list
    common_cols = list(set.intersection(*(set(df.columns) for df in frames)))
    combined_df = pd.concat([df[common_cols] for df in frames], ignore_index=True)
    duplicates = combined_df[combined_df.duplicated(subset =['AASeq'], keep='first')]
    duplicates = duplicates.sample(frac = 1, random_state = 1)
    return duplicates

print('Now splitting 5a12 data into PUL training & test sets...')
data_path = '../../data/5a12/'
#load sequences with single antigen binding status classification
vegf_pos = pd.read_csv(data_path + '5a12_vegf_pos_4th_sort.csv').drop_duplicates(subset='AASeq', keep='first')
vegf_neg = pd.read_csv(data_path + '5a12_vegf_neg_3rd_sort.csv').drop_duplicates(subset='AASeq', keep='first')
vegf_initial = pd.read_csv(data_path + '5a12_vegf_initial_sort.csv').drop_duplicates(subset='AASeq', keep='first')

vegf_pos['AgClass'] = 1
vegf_neg['AgClass'] = 0
vegf_initial['AgClass'] = 3

full_data = pd.concat([vegf_pos, vegf_neg, vegf_initial], ignore_index = True)
full_data = full_data.drop_duplicates(subset='AASeq', keep=False)

full_data, _ = consensus_seq_and_LD_to_df(data_frame = full_data, seq_col_str = 'AASeq', return_df = True)

for ed_thresh in [4,5,6,7,8]:
    
    thresh = ed_thresh
    train = full_data[full_data['LD'] <= thresh]
    test = full_data[full_data['LD'] > thresh]
    
    train = train[train['LD']>0]
    
    train_pos = train[(train['AgClass'] == 1)] 
    train_un = train[train['AgClass'] == 3]
    train_neg = train[train['AgClass'] == 0]
    
    #allocate 25% positive training sequences for use in validation and meta sets
    train_pos, train_pos_clean, _, _ = train_test_split(train_pos, train_pos['AgClass'], test_size = 0.25,
                                                              random_state = 1, shuffle = True, stratify = train_pos['LD'])
    
    train_pos_clean = train_pos_clean[train_pos_clean['LD']  != 1]
    #20 % total train seqs for validation set, 5 % total train seqs for meta
    val_pos, meta_pos, _, _ = train_test_split(train_pos_clean, train_pos_clean['AgClass'], test_size = 0.20,
                                                              random_state = 1, shuffle = True, stratify = train_pos_clean['LD'])
    
    val_neg, meta_neg, _, _ = train_test_split(train_neg, train_neg['AgClass'], test_size = 0.20,
                                                              random_state = 1, shuffle = True, stratify = train_neg['LD'])
    
    #Balance Classes
    def balance_classes_5a12_PUL(data_frame):
            df = data_frame.copy()
            positives = df[df['AgClass'] == 1]
            negatives  = df[df['AgClass'] == 0]
            truncate_value = min([len(ls) for ls in [positives, negatives]])
            
            if len(positives) > truncate_value:
                discarded_seqs, positives, _, _ = train_test_split(positives, positives['AgClass'], test_size = truncate_value,
                                                                          random_state = 1, shuffle = True, stratify = positives['LD'])
           
            if len(negatives) > truncate_value:
                negatives = negatives[negatives['LD'] != 0] #remove LD 0 for stratification on edit distance
                discarded_seqs, negatives, _, _ = train_test_split(negatives, negatives['AgClass'], test_size = truncate_value,
                                                                          random_state = 1, shuffle = True, stratify = negatives['LD'])   
            
            return pd.concat([positives,negatives],ignore_index = True).sample(frac = 1, random_state = 1)
        
        
    #set unlabeled class to negative
    train_un = train_un.copy()
    train_un['AgClass'] = 0
    train = balance_classes_5a12_PUL(data_frame = pd.concat([train_pos, train_un], ignore_index = True))
    
    val = pd.concat([val_pos, val_neg], ignore_index = True)
    meta = pd.concat([meta_pos, meta_neg], ignore_index = True)
    
    #Balance Classes
    test = balance_classes_5a12_PUL(data_frame = test)
    
    
    train.to_csv(data_path + f'{data_type}_ed_{ed_thresh}_train.csv', index=False, header=True)
    val.to_csv(data_path + f'{data_type}_ed_{ed_thresh}_val.csv', index=False, header=True)
    test.to_csv(data_path + f'{data_type}_ed_{ed_thresh}_test.csv', index=False, header=True)
    meta.to_csv(data_path + f'{data_type}_ed_{ed_thresh}_meta.csv', index=False, header=True)
    
