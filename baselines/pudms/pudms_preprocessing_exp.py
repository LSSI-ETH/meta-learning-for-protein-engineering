#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

data_type = '5a12_PUL_exp'

truncate_factor_list = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]

data_path = '../../data/5a12/'

test_set_out_path = 'test_sets/'

for edit_distance in [4,5,6,7]:
    for metaset in [0,1,2,3]:
        for trunc_factor in truncate_factor_list:
        
            train_pos_out_path = f'pos_data_edit_distance/ed_{edit_distance}/'
            train_un_out_path = f'un_data_edit_distance/ed_{edit_distance}/'
            
            train_df = pd.read_csv(data_path  + f'{data_type}_train_ed_{edit_distance}_truncated_{str(trunc_factor)}.csv') 
            meta_df = pd.read_csv(data_path  + f'{data_type}_meta_set_{metaset}_ed_{edit_distance}.csv') 
            
            train_df = pd.concat([train_df, meta_df], ignore_index=True)
            
            train_pos = train_df[train_df['AgClass'] == 1]
            train_un= train_df[train_df['AgClass'] == 0]
            train_pos = train_pos[['AASeq']]
            train_un = train_un[['AASeq']]
            
            train_pos.to_csv(train_pos_out_path + f'pos_ed_{edit_distance}_{str(trunc_factor)}_meta_{metaset}.txt', index=False, header = False)
            train_un.to_csv(train_un_out_path + f'un_ed_{edit_distance}_{str(trunc_factor)}_meta_{metaset}.txt', index=False, header = False)
    
    test_df  = pd.read_csv(data_path + f'{data_type}_ed_{edit_distance}_test.csv')
    test_pos = test_df[test_df['AgClass'] == 1]
    test_neg = test_df[test_df['AgClass'] == 0]
    
    test_pos = test_pos[['AASeq']]
    test_neg = test_neg[['AASeq']]
    
    
    test_pos.to_csv(test_set_out_path + f'test_ed_{edit_distance}_pos.txt', index=False, header = False)
    test_neg.to_csv(test_set_out_path + f'test_ed_{edit_distance}_neg.txt', index=False, header = False)
    
