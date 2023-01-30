#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef 
import argparse
import datetime
import os


parser = argparse.ArgumentParser(description='Meta Learning for Protein Engineering')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_classes', default='3', type=int,
                    help='number of class labels in data')
parser.add_argument('--trunc', default='0.75', type=float,
                    help='training data truncate factor (float) between 0 and 1')
parser.add_argument('--base_model', default='cnn', type=str,
                    help='base model to use, base, resnet, or densenet')
parser.add_argument('--data_type', default='4d5_syn', type=str,
                    help='library dataset, options: 4d5_syn, 4d5_exp, 5a12_PUL_syn, 5a12_PUL_exp, 5a12_2target')
parser.add_argument('--noise_fraction', default=0.0, type=float,
                   help='percent label noise to synthetically inject')
parser.add_argument('--learn_rate', default=5e-3, type=float,
                   help='initial optimizer learning rate')
parser.add_argument('--lr_scheduler', default=True, type=bool,
                   help='include learn rate scheduler')
parser.add_argument('--meta_size', default=96, type=int,
                   help='number of meta sequences to include ')
parser.add_argument('--dropout', default=0.3, type=float,
                   help='dropout fraction')
parser.add_argument('--vnet_lr', default=1e-4, type=float,
                   help='meta net learn rate')
parser.add_argument('--conv_filters', default=64, type=int,
                    help='number convolutional filters following transformer encoder')
parser.add_argument('--top_model', default='fine-tune', type=str,
                    help='model type. options: fine-tune, l2rw, mlc, metaset-only')
parser.add_argument('--opt_id', default='sgd', type=str,
                    help='options sgd, adam')
parser.add_argument('--noise_type', default='none', type=str,
                    help='none, flip')
parser.add_argument('--kernel', default=3, type=int,
                    help='size of 1D convolutional kernel')
parser.add_argument('--epochs', default=50, type=int,
                    help='number of training epochs')
parser.add_argument('--data_trunc_single_run', default='False', type=str,
                       help='runs model for only a single truncated data set. purpose: to optimize GPU use. options: True, False')
parser.add_argument('--mlc_k_steps', default=1, type=int,
                       help='mlc hyperparemeter k gradient steps')
parser.add_argument('--meta_set_number', default=1, type=int,
                       help='meta set index, options 0,1,2,3')
parser.add_argument('--evaluate_valset', default='False', type=str,
                       help='whether or not to evaluate metrics on validation set during training')
parser.add_argument('--alpha', default=0.0, type=float,
                       help='synthetic PUL fraction of positives in unlabeled set. range 0.1-0.8 in 0.1 intervals')

parser.set_defaults(augment=True)
args = parser.parse_args()

args.data_type = '5a12_PUL_exp'
truncate_factor_list = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]

data_path = 'predictions/'

def apply_class_label(row):
    
    if row['labeled'] == 0 and row['unlabeled'] == 1:
        return 0
    elif row['labeled'] == 1 and row['unlabeled'] == 0:
        return 1

meta_len_dict = {0: 32, 1: 96, 2: 288, 3: 864}

for order in [1]:
    
    args.base_model = 'pudms'
    args.top_model = f'pudms_order_{str(order)}'
    
    for metaset in [0,1,2,3]:
        
        for trunc_factor in truncate_factor_list:
                        
            hparams = {'data_type': args.data_type, 'basemodel': args.base_model, 'model': args.top_model, 
                            'noise_type': args.noise_type, 'noise_fraction': args.noise_fraction,
                            'train_len': trunc_factor, 'meta_len': meta_len_dict[metaset], 'seed': 0,
                            'meta_set': metaset,'optimizer': args.opt_id, 'learn_rate': args.learn_rate, 
                            'vnet_lr': args.vnet_lr, 'dropout': args.dropout,
                            'truncate_factor': trunc_factor, 'mlc_k_grad_steps': args.mlc_k_steps, 'alpha': 0.0}
                            
            #omit lowest data size for order 1 
            if metaset == 0 and trunc_factor == 0.01: continue    
            elif metaset == 1 and trunc_factor == 0.01: continue
        
            preds = pd.read_csv(data_path + f'test_preds_{str(trunc_factor)}_order_{str(order)}_meta_{metaset}.csv')
            
            preds['true_label'] = preds.apply (lambda row: apply_class_label(row), axis=1)
            preds['predicted_label'] = np.where(preds['pr_test'] >=0.5, 1, 0)
            
            y_true = preds['true_label'].to_list()
            y_pred = preds['predicted_label'].to_list()
            
            mcc =round(matthews_corrcoef(y_true, y_pred), 4)
            f1_micro = round(f1_score(y_true, y_pred, average='micro'), 4)
            
            print('==================')
            print(f'order = {str(order)}')
            print(f'trunc_factor = {trunc_factor}')
            print(f'mcc: {mcc}')
            print(f'metaset: {metaset}')
            print(f'f1_micro: {f1_micro}')
            
            metric_dict = {'output/best_f1': 0, 'output/best_mcc': 0,
                           'output/best_f1_epoch': 0, 'output/best_mcc_epoch': 0,
                           'output/final_val_mcc': 0, 'output/final_val_f1':0,
                           'output/test_mcc': mcc, 'output/test_f1': f1_micro}
            
            output_dict = {**hparams, **metric_dict}
            output_time = str(datetime.datetime.now().strftime("%d-%m-%Y_%H_%M"))
            output_dict['time'] = output_time
            
            filename = f'../../results/{args.data_type}_{args.base_model}.csv'
            df = pd.DataFrame.from_dict(output_dict, 'index').T.to_csv(filename, mode='a', index=False, header=(not os.path.exists(filename)))