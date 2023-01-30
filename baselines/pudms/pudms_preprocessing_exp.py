#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

data_type = '5a12_PUL'

truncate_factor_list = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]

data_path = '../../data/5a12/'
test_set_out_path = 'test_sets/'
train_pos_out_path = 'pos_data/'
train_un_out_path = 'un_data/'

for metaset in [0,1,2,3]:
    for trunc_factor in truncate_factor_list:
    
    
        train_df = pd.read_csv(data_path  + f'{data_type}_train_truncated_{str(trunc_factor)}.csv') 
        meta_df = pd.read_csv(data_path  + f'{data_type}_meta_set_{metaset}.csv') 
        
        train_df = pd.concat([train_df, meta_df], ignore_index=True)
        
        train_pos = train_df[train_df['AgClass'] == 1]
        train_un= train_df[train_df['AgClass'] == 0]
        train_pos = train_pos[['AASeq']]
        train_un = train_un[['AASeq']]
        
        train_pos.to_csv(train_pos_out_path + f'pos_{str(trunc_factor)}_meta_{metaset}.txt', index=False, header = False)
        train_un.to_csv(train_un_out_path + f'un_{str(trunc_factor)}_meta_{metaset}.txt', index=False, header = False)

test_df  = pd.read_csv(data_path + f'{data_type}_test.csv')
test_pos = test_df[test_df['AgClass'] == 1]
test_neg = test_df[test_df['AgClass'] == 0]

test_pos = test_pos[['AASeq']]
test_neg = test_neg[['AASeq']]


test_pos.to_csv(test_set_out_path + 'test_pos.txt', index=False, header = False)
test_neg.to_csv(test_set_out_path + 'test_neg.txt', index=False, header = False)

