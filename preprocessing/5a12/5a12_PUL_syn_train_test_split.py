#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing_utils import *
import numpy as np

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

print('Now splitting 5a12 data into synthetic PUL training & test sets...')
data_path = '../../data/5a12/'
#load sequences with single antigen binding status classification
vegf_pos = pd.read_csv(data_path + '5a12_vegf_pos_4th_sort.csv').drop_duplicates(subset='AASeq', keep='first')
vegf_neg = pd.read_csv(data_path + '5a12_vegf_neg_3rd_sort.csv').drop_duplicates(subset='AASeq', keep='first')

vegf_pos['AgClass'] = 1
vegf_neg['AgClass'] = 0

full_data = pd.concat([vegf_pos, vegf_neg], ignore_index = True)
full_data = full_data.drop_duplicates(subset='AASeq', keep=False)

full_data, _ = consensus_seq_and_LD_to_df(data_frame = full_data, seq_col_str = 'AASeq', return_df = True)

thresh = 5
train = full_data[full_data['LD'] <= thresh]
test = full_data[full_data['LD'] > thresh]

train = train[train['LD']>1]

train_pos = train[(train['AgClass'] == 1)] 
train_neg = train[train['AgClass'] == 0]

#20 % total train seqs for validation set, 5 % total train seqs for meta
train_pos, train_clean_pos, _, _ = train_test_split(train_pos, train_pos['AgClass'], test_size = 0.25,
                                                          random_state = 1, shuffle = True, stratify = train_pos['LD'])

val_pos, meta_pos, _, _ = train_test_split(train_clean_pos, train_clean_pos['AgClass'], test_size = 0.10,
                                                          random_state = 1, shuffle = True, stratify = train_clean_pos['LD'])


train_neg, train_clean_neg, _, _ = train_test_split(train_neg, train_neg['AgClass'], test_size = 0.10,
                                                          random_state = 1, shuffle = True, stratify = train_neg['LD'])

val_neg, meta_neg, _, _ = train_test_split(train_clean_neg, train_clean_neg['AgClass'], test_size = 0.10,
                                                          random_state = 1, shuffle = True, stratify = train_clean_neg['LD'])

val = pd.concat([val_pos, val_neg], ignore_index = True).sample(frac = 1, random_state = 1)
meta = pd.concat([meta_pos, meta_neg], ignore_index = True).sample(frac = 1, random_state = 1)

train = pd.concat([train_pos, train_neg], ignore_index = True).sample(frac = 1, random_state = 1)

data_type = '5a12_PUL_syn'

#Balance Classes
def balance_classes_pure(data_frame):
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

#Balance Classes
test = balance_classes_pure(data_frame = test)
train_pure = balance_classes_pure(data_frame = train)
#note do not use val set for metric evaluation / hyperparameter tuning as it will contain sequences present in training set
#only batch val set so as to not disrupt other data batching functions, which apply generally to other data sets
train_pure.to_csv(data_path + f'{data_type}_alpha_0.0_train.csv', index = False, header = True)
val.to_csv(data_path + f'{data_type}_val.csv', index = False, header = True)
meta.to_csv(data_path + f'{data_type}_meta.csv', index = False, header = True)
test.to_csv(data_path + f'{data_type}_test.csv', index = False, header = True)

def balance_synthetically_mixed_5a12_PUL(data_frame, alpha):
        df = data_frame.copy()
        positives = df[df['AgClass'] == 1]
        negatives  = df[df['AgClass'] == 0]
        false_negatives  = df[df['AgClass'] == 3]
        
        truncate_value = len(positives)
        neg_trunc = int((1-alpha) * truncate_value)
        false_neg_trunc = int(alpha * truncate_value) 
       
        if len(negatives) > (1-alpha) * truncate_value:
            negatives = negatives[negatives['LD'] != 0] #remove LD 0 for stratification on edit distance
            discarded_seqs, negatives, _, _ = train_test_split(negatives, negatives['AgClass'], test_size = neg_trunc,
                                                                      random_state = 1, shuffle = True, stratify = negatives['LD'])
            
        if len(false_negatives) > alpha * truncate_value + 1: #+1 for corner case where alpha * truncate_value = len(false_negatives) - 1
            
            discarded_seqs, false_negatives, _, _ = train_test_split(false_negatives, false_negatives['AgClass'], test_size = false_neg_trunc,
                                                                      random_state = 1, shuffle = True, stratify = false_negatives['LD'])
        
        return pd.concat([positives,negatives, false_negatives],ignore_index = True).sample(frac = 1, random_state = 1)

val_pos = val_pos.copy()
val_pos['AgClass'] = np.where(val_pos['AgClass'] == 1,3, val_pos['AgClass'])

for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:

    desired_number_positives = int(np.round(alpha * len(train_pos)))
    
    pos = train_pos.copy()
    neg = train_neg.copy()
    val_p = val_pos.copy()
    
    if alpha < 0.3: #where validation sequences can make up the required number of false positives, do not sample from positive set
        _, pos_for_neg, _, _ = train_test_split(val_p, val_p['AgClass'], test_size = desired_number_positives,
                                                                  random_state = 1, shuffle = True, stratify = val_p['LD'])
    else:
        desired_number_positives = desired_number_positives - len(val_p)
        if desired_number_positives > 0:
            pos, pos_for_neg, _, _ = train_test_split(pos, pos['AgClass'], test_size = desired_number_positives,
                                                                      random_state = 1, shuffle = True, stratify = pos['LD'])
            
        pos_for_neg = pd.concat([pos_for_neg, val_p], ignore_index=True).sample(frac = 1, random_state = 1)
        
    pos_for_neg = pos_for_neg.copy()
    pos_for_neg['AgClass'] = np.where(pos_for_neg['AgClass'] == 1, 3, pos_for_neg['AgClass'])
    
    train_out = pd.concat([pos, neg, pos_for_neg], ignore_index = True).sample(frac = 1, random_state = 1)
    train_out = balance_synthetically_mixed_5a12_PUL(train_out, alpha) 
    
    train_out.to_csv(data_path + f'{data_type}_alpha_{alpha}_train.csv', index = False, header = True)
