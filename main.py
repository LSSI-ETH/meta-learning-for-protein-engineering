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
from pathlib import Path
from config_args import *
from utils import *
from data_helper import *
from train_routines.basemodel import *
from train_routines.l2rw import *
from train_routines.mlc import *
from train_routines.metaset_only_baseline import *


def main(args): 

    #create required directories if necessary 
    Path(f'{args.checkpoint_path}').mkdir(parents=True, exist_ok=True)
    Path(f'{args.output_data_dir}').mkdir(parents=True, exist_ok=True)

    #-- param file cfg. used for checkpointing 
    args.param_file = f'{args.data_type}_{args.top_model}_{args.base_model}_{args.trunc}'

    if torch.cuda.is_available():
            use_cuda = True
            args.non_block = True
    else:
        use_cuda = False
        args.non_block = False
        
    device = torch.device("cuda" if use_cuda else "cpu")

    set_learn_rates(args)    

    if args.seed == 0: seed_list = [1,2,3]
    else: seed_list = [args.seed]

    for seed_entry in seed_list:
        torch.manual_seed(seed_entry)
        torch.cuda.manual_seed(seed_entry)
        np.random.seed(seed_entry)
        print(f'SEED {seed_entry}')     
        
        #==================== Instantiate Model & Optimizer ======================
        
        model_dict = {'fine-tune': BaseModel(args,device),
                    'l2rw': L2RW(args, device),
                    'mlc': MLC(args,device),
                    'metaset_baseline': MetaSetOnlyBaseline(args,device)}

        #==============  Loda Data, Encode Sequences ================
        
        if args.data_trunc_single_run: truncate_factor_list = [args.trunc]
        else: truncate_factor_list = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
        
        #====================    Restrict Train Size  =====================
        for trunc_factor in truncate_factor_list:
            
            train_loader, val_loader, test_loader, meta_loader = batch_datasets(args = args,
                                                                                truncate_factor = trunc_factor,
                                                                                seed_entry = seed_entry)
            
            args.train_len, args.meta_len, = len(train_loader.dataset), len(meta_loader.dataset)
            args.val_len, args.test_len  = len(val_loader.dataset), len(test_loader.dataset)
            args.truncate_factor = trunc_factor
            print('===============')
            print(f'Data Set = {args.data_type}')
            print(f'Truncate Factor = {trunc_factor}')
            print(f'\nNumber Training, Meta, Val, Test Samples: {args.train_len},'\
                f'{args.meta_len}, {args.val_len}, {args.test_len}')
            
            
            #====================           Initialize Model                              ======================
            model = model_dict[args.top_model]
            
            if torch.cuda.device_count() > 1 and args.top_model == 'fine-tune':
                print("Using", torch.cuda.device_count(), "GPUs!")
                model.model = nn.DataParallel(model.model)
            
            model.model.to(device)
            model_str = args.top_model

        
            print(f'\nNow Training {args.top_model}, with model {args.base_model}')
            print(f'learn rate, scheduler, optimizer:  {args.learn_rate},'\
                f'{args.lr_scheduler}, {args.opt_id}')
                
            
            # Check if checkpoints exists
            if not os.path.isfile(args.checkpoint_path + f'checkpoint_{args.param_file}.pth'):
                start_epoch, epoch = 0, 0
            else:    
                model.model, model.optimizer, start_epoch = load_checkpoint(model.model, 
                                                                                model.optimizer,
                                                                                device,
                                                                                args)
                epoch = start_epoch

            
            #====================== Tensorboard Initialization ===================
            current_time = str(datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
            
            if args.log_train_and_meta_metrics:
                run_name = f'runs/{args.data_type}_{args.base_model}_{args.top_model}_trunc_{trunc_factor}_{args.noise_type}_noise_{args.noise_fraction}_seed_{seed_entry}_{current_time}'
                writer = SummaryWriter(run_name)
                writer.flush()
            else:
                writer = None
                
            best_acc, best_acc_epoch, best_f1, best_f1_epoch, best_mcc, best_mcc_epoch = 0,0,0,0,0,0
            val_mcc, val_f1 = 0,0


            #account for async cuda operations if using gpu
            if torch.cuda.device_count() > 0:
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                start_time.record()
            else:
                training_start_time = time.time()           
                
            hparams = vars(args)
        
            fine_tune_epochs = 10

            #=================   Train & Eval Cycle    =====================
            for epoch in range(start_epoch, args.epochs):
                
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
                
                if trunc_factor >=0.25 and epoch % 5 == 0:
                    save_checkpoint(model.model, model.optimizer, epoch,
                                    device, args)
            
            #=============   Fine-Tuning Training  =====================
            if model_str == 'fine-tune':
                
                last_epoch = epoch
                for epoch in range(last_epoch + 1, last_epoch + fine_tune_epochs):
                    #print('Executing Fine Tuning on Meta Set...') 
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
            
            #account for async cuda operations if using gpu
            if torch.cuda.device_count() > 0:
                end_time.record()
                torch.cuda.synchronize()
                total_training_time = start_time.elapsed_time(end_time)/10**3
            else:                
                training_end_time = time.time()
                total_training_time = training_end_time - training_start_time
            
            print(f'best val mcc, f1 score = {best_mcc}, {best_f1}')
            test_f1, test_mcc  = model.test_step(test_loader, epoch, writer)
            print(f'test mcc, f1 = {test_mcc}, {test_f1}')

            #save final model weights
            with open(os.path.join(args.model_dir, f'{args.param_file}_model_final.pth'), 'wb') as f:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.model.state_dict(),
                    'optimizer_state_dict': model.optimizer.state_dict(),
                    }, f)

                
            metric_dict = {'output/best_f1': best_f1, 'output/best_mcc': best_mcc,
                        'output/best_f1_epoch': best_f1_epoch, 'output/best_mcc_epoch': best_mcc_epoch,
                        'output/final_val_mcc': val_mcc, 'output/final_val_f1':val_f1,
                        'output/test_mcc': test_mcc, 'output/test_f1': test_f1}
            
            if args.log_train_and_meta_metrics:
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
            output_dict['total_training_time'] = total_training_time
            
            filename = f'{args.output_data_dir}/{args.data_type}_{args.base_model}.csv'
            df = pd.DataFrame.from_dict(output_dict, 'index').T.to_csv(filename, mode='a', index=False, header=(not os.path.exists(filename)))
            del model

            #reinstantiate model option
            model_dict = {'fine-tune': BaseModel(args,device),
                        'l2rw': L2RW(args, device),
                        'mlc': MLC(args,device),
                        'metaset_baseline': MetaSetOnlyBaseline(args,device)}



if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)
    

    