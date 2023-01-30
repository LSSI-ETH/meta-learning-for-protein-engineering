#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import datetime
import argparse
import numpy as np
import re
import os
from utils import *
from data_helper import *
from train_routines.basemodel import *
from train_routines.l2rw import *
from train_routines.mlc import *
from train_routines.metaset_only_baseline import *

parser = argparse.ArgumentParser(description='Meta Learning for Machine Learning-Guided Protein Engineering')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_classes', default='3', type=int,
                    help='number of class labels in data')
parser.add_argument('--trunc', default='0.75', type=float,
                    help='training data truncate factor (float) between 0 and 1')
parser.add_argument('--base_model', default='cnn', type=str,
                    help='base model to use, base, resnet, or densenet')
parser.add_argument('--data_type', default='4d5_syn', type=str,
                    help='library dataset, options: 4d5_syn, 4d5_exp, 5a12_PUL_syn, 5a12_PUL_exp, 5a12_2ag')
parser.add_argument('--noise_fraction', default=0.0, type=float,
                   help='percent label noise to synthetically inject')
parser.add_argument('--learn_rate', default=5e-3, type=float,
                   help='initial optimizer learning rate')
parser.add_argument('--lr_scheduler', default=True, type=bool,
                   help='include learn rate scheduler')
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

if torch.cuda.is_available(): use_cuda = True
else: use_cuda = False

device = torch.device("cuda" if use_cuda else "cpu")

set_learn_rates(args)

if args.seed == 0: seed_list = [1,2,3]
else: seed_list = [args.seed]

for seed_entry in seed_list:
    torch.manual_seed(seed_entry)
    torch.cuda.manual_seed(seed_entry)
    np.random.seed(seed_entry)
    print(f'SEED {seed_entry}')     
    
    #====================          Instantiate Model & Optimizer                    ======================
     
    base_model = args.base_model

    params = {'data': args.data_type, 'model_name': base_model, 'lr': args.learn_rate, 'lr_scheduler': args.lr_scheduler,
              'p_dropout': args.dropout, 'vnet_lr': args.vnet_lr, 'conv_filters': args.conv_filters, 'opt': args.opt_id,
              'kernel': args.kernel, 'mlc_k_grad_steps': args.mlc_k_steps}
    
    
    model_dict = {'fine-tune': BaseModel(args,device),  
                  'l2rw': L2RW(args, device),
                  'mlc': MLC(args,device),
                  'metaset_baseline': MetaSetOnlyBaseline(args,device)}

    batch_size = args.batch_size

    #==============           Loda Data, Encode Sequences                      ================
    
    if args.data_trunc_single_run == 'True': truncate_factor_list = [args.trunc]
    else: truncate_factor_list = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
    
    def collect_len(loader):
        c = 0
        for i, (x,y) in enumerate(loader): c+= len(x)
        return c
    
    #====================            Restrict Train Size                                     =====================
    for trunc_factor in truncate_factor_list:
        
        train_loader, val_loader, test_loader, meta_loader = batch_datasets(args = args, truncate_factor = trunc_factor, seed_entry = seed_entry)
        
        train_len, meta_len, val_len, test_len = collect_len(train_loader), collect_len(meta_loader), collect_len(val_loader) , collect_len(test_loader)
        print('===============')
        print(f'Data Set = {args.data_type}')
        print(f'Truncate Factor = {trunc_factor}')
        print(f'\nNumber Training, Meta, Val, Test Samples: {train_len}, {meta_len}, {val_len}, {test_len}')
        
        
        #====================           Initialize Model                              ======================
        model = model_dict[args.top_model]
        model = model.to(device)
        model_str = args.top_model

        print(f'\nNow Training {args.top_model}, with model {args.base_model}')
        print(f'learn rate, scheduler, optimizer:  {args.learn_rate}, {args.lr_scheduler}, {args.opt_id}')

        
        #====================== Tensorboard Initialization ===================================
        current_time = str(datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
        
        run_name = f'runs/{args.data_type}_{args.base_model}_{args.top_model}_trunc_{trunc_factor}_{args.noise_type}_noise_{args.noise_fraction}_seed_{seed_entry}_{current_time}'
        writer = SummaryWriter(run_name)
        writer.flush()
        
        best_acc, best_acc_epoch, best_f1, best_f1_epoch, best_mcc, best_mcc_epoch = 0,0,0,0,0,0
        val_mcc, val_f1 = 0,0
                        
        hparams = {'data_type': args.data_type, 'basemodel': args.base_model, 'model': args.top_model, 
                   'noise_type': args.noise_type, 'noise_fraction': args.noise_fraction,
                   'train_len': train_len, 'meta_len': meta_len, 'seed': seed_entry,
                   'meta_set': args.meta_set_number,'optimizer': args.opt_id, 'learn_rate': args.learn_rate, 
                   'vnet_lr': args.vnet_lr, 'dropout': args.dropout,
                   'truncate_factor': trunc_factor, 'mlc_k_grad_steps': args.mlc_k_steps, 'alpha': args.alpha}
    
        
        fine_tune_epochs = 10

        #======================     Train & Eval Cycle          ===================================
        for epoch in range(args.epochs):
            fine_tune = False
            model.train_step(train_loader,meta_loader,epoch,writer,args.batch_size)
            if args.evaluate_valset == 'True':
                val_f1, val_mcc = model.test_step(val_loader, epoch, writer)
    
                #record best performance
                if val_f1 >= best_f1:
                    best_f1 = val_f1
                    best_f1_epoch = epoch
                if val_mcc >= best_mcc:
                    best_mcc = val_mcc
                    best_mcc_epoch = epoch

        
         #======================     Fine-Tuning Training        ==========================
        if model_str == 'fine-tune':
            fine_tune = True
            last_epoch = epoch
            for epoch in range(last_epoch + 1, last_epoch + fine_tune_epochs):
                print('Executing Fine Tuning on Meta Set...') 
                model.train_step(meta_loader,meta_loader,epoch,writer,args.batch_size) 
                if args.evaluate_valset == 'True':
                    val_f1, val_mcc = model.test_step(val_loader, epoch, writer)
                    #record best performance
                    if val_f1 >= best_f1:
                        best_f1 = val_f1
                        best_f1_epoch = epoch
                        
                    if val_mcc >= best_mcc:
                        best_mcc = val_mcc
                        best_mcc_epoch = epoch
                
            
            print(f'best fine-tune val f1 score = {best_f1}')
            print(f'best fine-tune val mcc = {best_mcc}')
        
        print(f'best val mcc, f1 score = {best_mcc}, {best_f1}')
        test_f1, test_mcc  = model.test_step(test_loader, epoch, writer)
        print(f'test mcc, f1 = {test_mcc}, {test_f1}')
        
        metric_dict = {'output/best_f1': best_f1, 'output/best_mcc': best_mcc,
                       'output/best_f1_epoch': best_f1_epoch, 'output/best_mcc_epoch': best_mcc_epoch,
                       'output/final_val_mcc': val_mcc, 'output/final_val_f1':val_f1,
                       'output/test_mcc': test_mcc, 'output/test_f1': test_f1}
        
        writer.add_hparams(hparams, metric_dict = metric_dict)
        writer.flush()
        writer.close()
        
        print('Now Deleting Model')
        non_tensor_keys = ['output/best_mcc_epoch', 'output/best_f1_epoch']
        for key, value in metric_dict.items():
            if key not in non_tensor_keys:
                m = re.search(r'\((.*)\)', str(value))                                
                try:
                    metric_dict[key] = m.group(1)
                except:
                    pass
                    
        output_dict = {**hparams, **metric_dict}
        output_time = str(datetime.datetime.now().strftime("%d-%m-%Y_%H_%M"))
        output_dict['time'] = output_time
        
        filename = f'results/{args.data_type}_{args.base_model}.csv'
        df = pd.DataFrame.from_dict(output_dict, 'index').T.to_csv(filename, mode='a', index=False, header=(not os.path.exists(filename)))
        del model

        #reinstantiate model option
        model_dict = {'fine-tune': BaseModel(args,device),  
      'l2rw': L2RW(args, device),
      'mlc': MLC(args,device),    
      'metaset_baseline': MetaSetOnlyBaseline(args,device)}