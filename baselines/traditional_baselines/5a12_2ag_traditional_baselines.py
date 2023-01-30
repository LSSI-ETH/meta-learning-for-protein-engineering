#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import argparse
import datetime
import os
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

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

parser.set_defaults()
args = parser.parse_args()   

data_type = '5a12_2ag'
args.data_type = data_type
data_path = '../../data/5a12/'

def onehot(aa_seqs):
    '''
    one-hot encoding of a list of amino acid sequences with padding
    parameters:
        - aa_seqs : list with CDR3 sequences
    returns:
        - enc_aa_seq : list of np.ndarrays containing padded, encoded amino acid sequences
    '''
    ### Create an Amino Acid Dictionary
    aa_list = sorted(['A', 'C', 'D', 'E','F','G', 'H','I', 'K', 'L','M','N', \
              'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'UNK'])
    
    aa_dict = {char : l for char, l in zip(aa_list, np.eye(len(aa_list), k=0))}
    
    #####pad the longer sequences with '-' sign
    #1) identify the max length
    max_seq_len = max([len(x) for x in aa_seqs])
    #2) pad the shorter sequences with '-'
    aa_seqs = [seq + (max_seq_len - len(seq))*'-'
                      for i, seq in enumerate(aa_seqs)]
    
    # encode sequences:
    sequences=[]
    for seq in aa_seqs:
        e_seq=np.zeros((len(seq),len(aa_list)))
        count=0
        for aa in seq:
            if aa in aa_list:
                e_seq[count]=aa_dict[aa]
                count+=1
            else:
                print ("Unknown amino acid in peptides: "+ aa +", encoding aborted!\n")
        sequences.append(e_seq)
    enc_aa_seq = np.asarray(sequences)
    #flatten seqeunces
    enc_aa_seq = np.reshape(enc_aa_seq, (enc_aa_seq.shape[0], -1))
    return enc_aa_seq

test_df = pd.read_csv(data_path  + f'{data_type}_test.csv') 
X_test = onehot(test_df['AASeq'].to_list())
y_test =  np.asarray(test_df['AgClass'])

truncate_factor_list = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
classifier_list = ['RF', 'SVM', 'NB']

for classifier_str in classifier_list:
    
    args.base_model = 'traditional_baseline'
    args.top_model = f'{classifier_str}'
    
    for metaset in [0,1,2,3]:
    
        for seed in [1,2,3]:
            
            f1_scores = []
            mcc_scores = []
            
            for trunc_factor in truncate_factor_list:
            
                meta_df = pd.read_csv(data_path  + f'{data_type}_meta_set_{metaset}.csv') 
                train_df = pd.read_csv(data_path  + f'{data_type}_train_truncated_{str(trunc_factor)}.csv') 

                hparams = {'data_type': args.data_type, 'basemodel': args.base_model, 'model': args.top_model, 
                'noise_type': args.noise_type, 'noise_fraction': args.noise_fraction,
                'train_len': len(train_df), 'meta_len': len(meta_df), 'seed': seed,
                'meta_set': metaset,'optimizer': args.opt_id, 'learn_rate': args.learn_rate, 
                'vnet_lr': args.vnet_lr, 'dropout': args.dropout,
                'truncate_factor': trunc_factor, 'mlc_k_grad_steps': args.mlc_k_steps, 'alpha': 0.0}
                
                train_df = pd.concat([train_df,meta_df],ignore_index=True)
                
                X_train = onehot(train_df['AASeq'].to_list())
                y_train = np.asarray(train_df['AgClass'])
                
                print("-------------------")
                print(f'Meta set = {metaset}')
                print(f'Truncate Factor = {trunc_factor}')
                print(f"Classifier = {classifier_str}")
                
                if classifier_str == 'RF':
                    estimator = RandomForestClassifier(
                        n_estimators=200,
                        criterion='gini',
                        bootstrap=True,
                        n_jobs=1,
                        random_state = seed
                    )
                elif classifier_str == 'NB':
                    estimator = GaussianNB()
                elif classifier_str == 'SVM':
                    estimator = LinearSVC(random_state= seed)
                    
                estimator.fit(X_train, y_train)
                y_pred = estimator.predict(X_test)
                
                mcc =round(matthews_corrcoef(y_test, y_pred), 4)
                f1_micro = round(f1_score(y_test, y_pred, average='micro'), 4)
                f1_scores.append(f1_micro)
                mcc_scores.append(mcc)
                print(f'F1 Micro: {f1_micro}')
                print(f'MCC: {mcc}')
                
                metric_dict = {'output/best_f1': 0, 'output/best_mcc': 0,
                            'output/best_f1_epoch': 0, 'output/best_mcc_epoch': 0,
                            'output/final_val_mcc': mcc, 'output/final_val_f1':f1_micro,
                            'output/test_mcc': mcc, 'output/test_f1': f1_micro}
                
                output_dict = {**hparams, **metric_dict}
                output_time = str(datetime.datetime.now().strftime("%d-%m-%Y_%H_%M"))
                output_dict['time'] = output_time
                
                filename = f'../../results/{args.data_type}_{args.base_model}.csv'
                df = pd.DataFrame.from_dict(output_dict, 'index').T.to_csv(filename, mode='a', index=False, header=(not os.path.exists(filename)))
                
