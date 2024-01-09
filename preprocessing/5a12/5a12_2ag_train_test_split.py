#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from preprocessing_utils import *

def combine_df_list_and_shuffle(df_list, keep = False):
    '''
    combines two dataframes, drops duplicates, & shuffles
    
    Parameters
    ----------

    data_frame : pandas.DataFrame
        dataframe containing all sequence & label data
    keep: bool
        whether or not to keep duplicates

    Returns
    -------
    data_frame : pandas.DataFrame
        combined, shuffled dataframe
    '''
    frames = df_list
    common_cols = list(set.intersection(*(set(df.columns) for df in frames)))
    combined_df = pd.concat([df[common_cols] for df in frames], ignore_index=True).drop_duplicates(subset='AASeq', keep=keep)
    combined_df = combined_df.sample(frac = 1, random_state = 1)
    return combined_df

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
    #duplicates = pd.DataFrame()
    duplicates = combined_df[combined_df.duplicated(subset =['AASeq'], keep='first')]
    duplicates = duplicates.sample(frac = 1, random_state = 1)
    return duplicates


def class_bal_df(df, truncation_multiple_factor = 2):
    double_pos = df[df['AgClass'] == 3]
    double_neg = df[df['AgClass'] == 2]
    v_pos_a_neg = df[df['AgClass'] == 1]
    v_neg_a_pos = df[df['AgClass'] == 0]

    v_neg_a_pos = v_neg_a_pos[v_neg_a_pos['LD'] != 1] #drop LD1 from df to facilitate splitting on LD
    double_neg = double_neg[double_neg['LD'] != 1] #drop LD1 from df to facilitate splitting on LD

    min_list = min([len(ls) for ls in [double_pos, double_neg, v_pos_a_neg,v_neg_a_pos]])
    truncate_value = int(np.round(truncation_multiple_factor*min_list))
    
    if len(double_pos) > truncate_value: 
        discarded_seqs, double_pos, y_train, y_discard = train_test_split(double_pos, double_pos['AgClass'], test_size = truncate_value,
                                                                  random_state = 1, shuffle = True, stratify = double_pos['LD'])
    if len(double_neg) > truncate_value + 100: #add smoothing factor to enable train/test splitting on smaller datasets
        
        discarded_seqs, double_neg, y_train, y_discard = train_test_split(double_neg, double_neg['AgClass'], test_size = truncate_value,
                                                                  random_state = 1, shuffle = True, stratify = double_neg['LD'])
    if len(v_pos_a_neg) > truncate_value + 100 :#add smoothing factor to enable train/test splitting on smaller datasets
        discarded_seqs, v_pos_a_neg, y_train, y_discard = train_test_split(v_pos_a_neg, v_pos_a_neg['AgClass'], test_size = truncate_value,
                                                                  random_state = 1, shuffle = True, stratify = v_pos_a_neg['LD'])
        
    if len(v_neg_a_pos) > truncate_value + 100: #add smoothing factor for edit distance case wehre v_neg_a_pos = truncate_faclue + 1
        
        discarded_seqs, v_neg_a_pos, y_train, y_discard = train_test_split(v_neg_a_pos, v_neg_a_pos['AgClass'], test_size = truncate_value,
                                                                  random_state = 1, shuffle = True, stratify = v_neg_a_pos['LD'])
    
    out_df = combine_df_list_and_shuffle([double_pos, double_neg, v_pos_a_neg, v_neg_a_pos], keep = False)
    return out_df
    
def shuffle_and_split_into_two(df):
    #arbitrarily split positive data & assign feaux double pos/neg or mixed pos/neg labels
    df = df.sample(frac=1, random_state = 1)
    np.random.seed(0)
    split_array = np.array_split(df, 2)  
    split_1 = split_array[0].copy()
    split_2 = split_array[1].copy()
    return split_1, split_2


print('Now splitting 5a12 data into 2ag training & test sets...')
data_path = '../../data/5a12/'
#load sequences with single antigen binding status classification
vegf_pos = pd.read_csv(data_path + '5a12_vegf_pos_4th_sort.csv').drop_duplicates(subset='AASeq', keep='first')
vegf_neg = pd.read_csv(data_path + '5a12_vegf_neg_3rd_sort.csv').drop_duplicates(subset='AASeq', keep='first')
ang2_pos = pd.read_csv(data_path + '5a12_ang2_pos_2nd_sort.csv').drop_duplicates(subset='AASeq', keep='first')
ang2_neg = pd.read_csv(data_path + '5a12_ang2_neg_2nd_sort.csv').drop_duplicates(subset='AASeq', keep='first')

vegf_pos['df'], vegf_neg['df'], ang2_pos['df'], ang2_neg['df'] = 'vp', 'vn', 'ap', 'an' 
#allocat full single antigen data sets
vegf_only = combine_df_list_and_shuffle( [vegf_pos.copy(), vegf_neg.copy()], keep = False)
ang2_only = combine_df_list_and_shuffle([ang2_pos.copy(), ang2_neg.copy()], keep = False)

#allocate sequences present in multiple pools
double_pos = combine_df_and_retain_duplicates([vegf_only[vegf_only['df'] == 'vp'].copy(), ang2_only[ang2_only['df'] == 'ap'].copy()])
double_neg = combine_df_and_retain_duplicates([vegf_only[vegf_only['df'] == 'vn'].copy(), ang2_only[ang2_only['df'] == 'an'].copy()])
v_pos_a_neg = combine_df_and_retain_duplicates([vegf_only[vegf_only['df'] == 'vp'].copy(), ang2_only[ang2_only['df'] == 'an'].copy()])
v_neg_a_pos = combine_df_and_retain_duplicates([vegf_only[vegf_only['df'] == 'vn'].copy(), ang2_only[ang2_only['df'] == 'ap'].copy()])

#sequences with only one label. i.e. not present in multiple pools
single_label_seqs = combine_df_list_and_shuffle([vegf_only.copy(), ang2_only.copy()], keep=False) 
vegf_pos_only = single_label_seqs[single_label_seqs['df'] == 'vp'].copy()
vegf_neg_only = single_label_seqs[single_label_seqs['df'] == 'vn'].copy()
ang2_pos_only = single_label_seqs[single_label_seqs['df'] == 'ap'].copy()
ang2_neg_only = single_label_seqs[single_label_seqs['df'] == 'an'].copy()

#assign double & single antigen class labels
v_neg_a_pos['AgClass'] = 0
v_pos_a_neg['AgClass'] = 1
double_neg['AgClass'] = 2
double_pos['AgClass'] = 3
vegf_pos_only['AgClass'] = 4
vegf_neg_only['AgClass'] = 5
ang2_pos_only['AgClass'] = 6
ang2_neg_only['AgClass'] = 7

data_frame = [v_neg_a_pos.copy(), v_pos_a_neg.copy(), double_neg.copy(), double_pos.copy(), 
              vegf_pos_only.copy(), vegf_neg_only.copy(), ang2_pos_only.copy(), ang2_neg_only.copy()]

data_frame = combine_df_list_and_shuffle(data_frame, keep=False) #shuffle & drop duplicates

data_frame, _ = consensus_seq_and_LD_to_df(data_frame = data_frame, seq_col_str = 'AASeq', return_df = True)
#split train & test set by edit distance, class balance
for ed_thresh in [4,5,6,7,8]:
    
    train_test_LD_split_threshold = ed_thresh
    train_df = data_frame[data_frame['LD'] <= train_test_LD_split_threshold]
    test_df = data_frame[data_frame['LD'] > train_test_LD_split_threshold]
    
    #retain only sequences that are classified for both antigens in test set
    test_df = test_df[(test_df['AgClass'] == 0) | (test_df['AgClass'] == 1) | (test_df['AgClass'] == 2) | (test_df['AgClass'] == 3)]
    #break up training set into single & double classified training sets
        #double classified will be used for meta data set
        #single classified will be used for training
    train_df_double_classified = train_df[(train_df['AgClass'] == 0) | (train_df['AgClass'] == 1) | (train_df['AgClass'] == 2) | (train_df['AgClass'] == 3)]
    train_df_vegf_classified_only = train_df[(train_df['AgClass'] == 4) | (train_df['AgClass'] == 5)]
    train_df_ang2_classified_only = train_df[(train_df['AgClass'] == 6) | (train_df['AgClass'] == 7)]
    
    #allocate 20% of double labeled training  set for validation if needed for hyperparameter tuning
    train_df_double_classified, val_df, y_train, y_val = train_test_split(train_df_double_classified, train_df_double_classified['AgClass'], test_size = 0.2,
                                                              random_state = 1, shuffle = True, stratify = train_df_double_classified['LD'])
    
    
    
    #============================== Batch 5a12 binary to multiclass    ======================
    #def batch_5a12_binary_to_multi_antigen(args): 
        
    def df_to_x_y(df):
        x, y = df['AASeq'], df['AgClass']
        return x,y
    
    #class key
    #v_neg_a_pos['AgClass'], v_pos_a_neg['AgClass'] = 0, 1
    #double_neg['AgClass'], double_pos['AgClass'] = 2, 3
    #vegf_pos_only['AgClass'], vegf_neg_only['AgClass'] = 4,5 
    
    #========================      Batch Training Set       ==================================
    #using sequences labeled for vegf, randomly assign an Ang2 label
    ang2_pos_train = train_df_vegf_classified_only[train_df_vegf_classified_only['AgClass'] == 4]
    ang2_neg_train = train_df_vegf_classified_only[train_df_vegf_classified_only['AgClass'] == 5]
    
    double_pos_train, vegf_pos_ang2_neg_train = shuffle_and_split_into_two(ang2_pos_train)
    double_neg_train, vegf_neg_ang2_pos_train = shuffle_and_split_into_two(ang2_neg_train)
    
    double_pos_train['AgClass'] = 3
    double_neg_train['AgClass'] = 2
    vegf_pos_ang2_neg_train['AgClass'] = 1
    vegf_neg_ang2_pos_train['AgClass'] = 0
    
    train_df_list = [double_pos_train.copy(), double_neg_train.copy(), vegf_pos_ang2_neg_train.copy(), vegf_neg_ang2_pos_train.copy()]
    train_df = combine_df_list_and_shuffle(train_df_list, keep = False)

    train_df = class_bal_df(train_df, truncation_multiple_factor = 1)
    test_df = class_bal_df(test_df, truncation_multiple_factor = 16)
    val_df = class_bal_df(val_df, truncation_multiple_factor = 16)
    
    data_path = '../../data/5a12/'
    data_type = '5a12_2ag'
    train_df.to_csv(data_path + f'{data_type}_ed_{ed_thresh}_train.csv', index=False, header=True)
    train_df_double_classified.to_csv(data_path + f'{data_type}_ed_{ed_thresh}_clean_train.csv', index=False, header=True)
    val_df.to_csv(data_path + f'{data_type}_ed_{ed_thresh}_val.csv', index=False, header=True)
    test_df.to_csv(data_path + f'{data_type}_ed_{ed_thresh}_test.csv', index=False, header=True)
    