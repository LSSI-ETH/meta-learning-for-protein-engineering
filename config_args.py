#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse


def create_parser():
    parser = argparse.ArgumentParser(description='Meta Learning for Machine Learning-Guided Protein Engineering',
                                     fromfile_prefix_chars='@')
    
    # path
    parser.add_argument('--train_path', type=str, default='data/')
    parser.add_argument('--output_data_dir', type=str, default='results/')
    parser.add_argument('--model_dir', type=str, default='model_weights/')
    parser.add_argument('--checkpoint_path', type=str, default='model_weights/')
    parser.add_argument('--param_file', type=str, default='none')
    
    # training
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', default=50, type=int,
                        help='number of training epochs')
    parser.add_argument('--num_classes', default='3', type=int,
                        help='number of class labels in data')
    parser.add_argument('--trunc', default='0.75', type=float,
                        help='training data truncate factor (float) between 0 and 1')
    parser.add_argument('--opt_id', default='sgd', type=str,
                        choices=['sgd', 'adam'])
    parser.add_argument('--learn_rate', default=5e-3, type=float,
                    help='initial optimizer learning rate')
    parser.add_argument('--lr_scheduler', default=True, type=bool,
                    help='include learn rate scheduler')
    parser.add_argument('--data_trunc_single_run', action='store_true',
                        help='runs model for only a single truncated data set.'\
                            ' purpose: to optimize GPU use. options:'\
                                ' True, False')
    
    # dataset
    parser.add_argument('--data_type', default='4d5_syn', type=str,
                        choices=['4d5_syn', '4d5_exp', '5a12_PUL_syn', 
                                 '5a12_PUL_exp', '5a12_2ag'])
    parser.add_argument('--meta_set_number', default=1, type=int,
                        help='meta set index, options 0,1,2,3')
    parser.add_argument('--edit_distance', default=-1, type=int,
                        help='train/test edit distance split valid between'\
                            '4 and 7 for experimental tasks.-1 is default split'\
                                'used for synthetic tasks')
    
    # base model & meta model
    parser.add_argument('--base_model', default='cnn', type=str,
                        choices=['cnn', 'transformer',
                                 'logistic_regression', 'mlp'])
    parser.add_argument('--top_model', default='fine-tune', type=str,
                        choices=['standard', 'fine-tune', 'l2rw', 'mlc', 
                                 'metaset_baseline'])
    parser.add_argument('--mlc_hdim', default=32, type=int,
                        help='hiddem dim of MLC LCN')
    parser.add_argument('--vnet_lr', default=1e-4, type=float,
                    help='meta net learn rate')
    parser.add_argument('--mlc_k_steps', default=1, type=int,
                        help='mlc hyperparemeter k gradient steps')
    
    # model hyperparameters
    parser.add_argument('--conv_filters', default=64, type=int,
                        help='number convolutional filters following transformer encoder')
    parser.add_argument('--dropout', default=0.3, type=float,
                    help='dropout fraction')
    parser.add_argument('--kernel', default=3, type=int,
                        help='size of 1D convolutional kernel')
    
    # noisy training
    parser.add_argument('--noise_fraction', default=0.0, type=float,
                    help='percent label noise to synthetically inject')
    parser.add_argument('--noise_type', default='none', type=str,
                        help='none, flip')
    parser.add_argument('--alpha', default=0.0, type=float,
                        help='synthetic PUL fraction of positives in unlabeled'\
                            'set. range 0.1-0.8 in 0.1 intervals')
    
    
    # metrics 
    parser.add_argument('--evaluate_valset', action='store_true',
                        help='whether or not to evaluate metrics on'\
                            'validation set during training')
    parser.add_argument('--log_train_and_meta_metrics', action='store_true',
                    help='enbale tensorboard logging & compute training'\
                        ' and meta metrics during training. Default = False')
    parser.add_argument('--non_block', action='store_true',
                    help='controls non_blocking & pinned memory')
    parser.set_defaults(augment=True)

    return parser