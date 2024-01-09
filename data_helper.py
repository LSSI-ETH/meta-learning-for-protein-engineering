#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from utils import *
from train_routines  import *


def batch_datasets(args, truncate_factor, seed_entry):
    
    path_5a12 = 'data/5a12/'
    path_4d5 = 'data/4d5/'
    if args.data_type.startswith('5a12'): data_path = path_5a12
    elif args.data_type.startswith('4d5'): data_path = path_4d5

        
    if 'syn' in args.data_type:
        
        if args.data_type == '5a12_PUL_syn':
            train_df = pd.read_csv(data_path  + f'{args.data_type}_alpha_{args.alpha}_train_truncated_{str(truncate_factor)}.csv') 
        else:
            train_df = pd.read_csv(data_path  + f'{args.data_type}_train_truncated_{str(truncate_factor)}.csv') 
            
        val_df  = pd.read_csv(data_path + f'{args.data_type}_val.csv')        
        test_df  = pd.read_csv(data_path + f'{args.data_type}_test.csv')
        meta_df = pd.read_csv(data_path  + f'{args.data_type}_meta_set_{str(args.meta_set_number)}.csv')
    
    else:
        
        assert args.edit_distance != -1, 'Please supply edit_distance argument for experimental learning tasks;'\
             ' an integer between 4 and 7, i.e. --edit_distance=4'

        train_df = pd.read_csv(data_path  + f'{args.data_type}_train_ed_{args.edit_distance}_truncated_{str(truncate_factor)}.csv') 
        val_df  = pd.read_csv(data_path + f'{args.data_type}_ed_{args.edit_distance}_val.csv')        
        test_df  = pd.read_csv(data_path + f'{args.data_type}_ed_{args.edit_distance}_test.csv')
        meta_df = pd.read_csv(data_path  + f'{args.data_type}_meta_set_{str(args.meta_set_number)}_ed_{args.edit_distance}.csv')
        
    def x_y_split(df, args):
        x,y = df['AASeq'], df['AgClass']
        return x, y
    
    x_train, y_train = x_y_split(df = train_df, args = args)
    x_val, y_val = x_y_split(df = val_df, args = args)
    x_test, y_test = x_y_split(df = test_df, args = args)
    x_meta, y_meta = x_y_split(df = meta_df, args = args)
    
    x_train = encode_ngrams(x_train, args)
    x_val = encode_ngrams(x_val, args)
    x_test = encode_ngrams(x_test, args)
    x_meta = encode_ngrams(x_meta, args)
    
    if args.data_type == '4d5_syn':
        y_train = noise_injection(args, y_train, seed_entry)
    
    train_loader, val_loader, test_loader, meta_loader  = data_to_loader(x_train = x_train, x_meta = x_meta, 
                                                                         x_val = x_val, x_test = x_test, 
                                                                         y_train = y_train, y_val = y_val,
                                                                         y_test = y_test, y_meta = y_meta, args = args)
                                                                         
    return train_loader, val_loader, test_loader, meta_loader