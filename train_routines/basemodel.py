#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import random
import numpy as np
import higher
from torch.utils.tensorboard import SummaryWriter
import datetime
import copy
from models import *
import itertools
import time
from torchmetrics import F1, MatthewsCorrcoef


class DummyScheduler(torch.optim.lr_scheduler._LRScheduler):
    '''
    dummy scheduler for use with MLC algorithm. collects learning rate per parameter
    '''
    def get_last_lr(self):
        lrs = []
        for param_group in self.optimizer.param_groups:
            lrs.append(param_group['lr'])
                       
        return lrs

    def step(self, epoch=None):
        pass


# ---------------------- Base Model Trainer ----------------------
class BaseModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()

        self.device = device
        self.args = args
        
        if '4d5' in args.data_type: self.num_classes = 3
        elif args.data_type == '5a12_2ag': self.num_classes = 4
        elif args.data_type.startswith('5a12_PUL'): self.num_classes = 2
        
        self.input_size = 17 #input sequence length
        self.ntokens = 21 #transformer embedding vocabulary
        
        #base model selection
        if args.base_model == 'cnn':
            filters, dense  = 64, 512
            self.model = CNN(input_size = self.input_size, 
                             conv_filters = filters, 
                             dense_nodes = dense,
                             n_out = self.num_classes,
                             kernel_size = args.kernel, 
                             dropout = args.dropout
                             ).to(self.device)

        elif args.base_model == 'transformer':
            self.model = Transformer(ntoken = self.ntokens,
                                       emb_dim = 32,
                                       nhead = 2,
                                       nhid = 128,
                                       nlayers = 1,
                                       n_classes = self.num_classes,
                                       seq_len = self.input_size,
                                       dropout = args.dropout,
                                       out_dim = 512
                                       ).to(self.device)

        elif args.base_model == 'logistic_regression':
            self.model = LogisticRegression(input_size = self.input_size,
                                            n_classes = self.num_classes
                                            ).to(self.device)
        
        elif args.base_model == 'mlp':
            self.model = MLP(input_size = self.input_size,
                             hdim1 = 256,
                             hdim2 = 512, 
                             n_out = self.num_classes,
                             dropout = args.dropout
                             ).to(self.device)

        #learning rate, & lr schedule, loss fn
        self.learning_rate = args.learn_rate

        if args.opt_id == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=self.learning_rate,
                                              amsgrad=True, eps = 1e-8)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=self.learning_rate,
                                             momentum=0.9)

        self.lr_scheduler = args.lr_scheduler
        
        if self.lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                  milestones=[15,30],
                                                                  gamma=0.1)
        else: 
            if args.top_model == 'mlc':
                self.scheduler = DummyScheduler(self.optimizer)
            else:
                self.scheduler = None
        
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)

    def get_predictions(self,model, X, args, mask = None):

        if 'transformer' in args.base_model:        
            pred = model(X)
        else:       
            pred = model(X.float())
        return pred

    def mlc_get_predictions(self,model, X, args):
        
        if 'transformer' in args.base_model:        
            pred = model(X, return_h=True)
        else:
            pred = model(X.float(), return_h=True)
        return pred
    

    def train_step(self,train_dataloader,meta_loader, epoch, tensorboard_writer, batch_size):
        train_loss = 0
        f1 = F1(num_classes = self.num_classes).to(self.device)
        mcc = MatthewsCorrcoef(num_classes = self.num_classes).to(self.device)
        
        self.model.train()
        for batch, (X, labels) in enumerate(train_dataloader):
            X = X.to(self.device, non_blocking = self.args.non_block)
            labels = labels.to(self.device, non_blocking = self.args.non_block)
            pred = self.get_predictions(self.model, X, self.args)                
            loss = self.loss_fn(pred,labels)

            if self.args.log_train_and_meta_metrics:
                train_loss = loss.item()
                tensorboard_writer.add_scalar('train/ Loss', train_loss, epoch * len(train_dataloader) + batch)
                
            self.optimizer.zero_grad() 
            loss.backward()        
            
            self.optimizer.step()
            
            if self.scheduler is not None: self.scheduler.step()            
        
            if self.args.log_train_and_meta_metrics:
                with torch.no_grad():
                    predicted = (F.softmax(pred,1).data.argmax(1))
                    batch_f1_mtr = f1(predicted,labels) #open bug in torchmetrics F1, therefore, softmax must be applied prior to calc
                    batch_mcc_mtr = mcc(pred,labels)

        if self.args.log_train_and_meta_metrics:                    
            epoch_f1 = f1.compute()
            epoch_mcc = mcc.compute()

        if self.args.log_train_and_meta_metrics:
            tensorboard_writer.add_scalar('train/ F1', epoch_f1, epoch)
            tensorboard_writer.add_scalar('train/ MCC', epoch_mcc, epoch)
            tensorboard_writer.flush()

                
    def test_step(self, test_loader,epoch, tensorboard_writer):
        self.model.eval()
        test_loss = 0
        f1 = F1(num_classes = self.num_classes, compute_on_step=False).to(self.device)
        mcc = MatthewsCorrcoef(num_classes = self.num_classes, compute_on_step=False).to(self.device)
        
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(test_loader):
                inputs = inputs.to(self.device, non_blocking = self.args.non_block)
                labels = labels.to(self.device, non_blocking = self.args.non_block)
                outputs = self.get_predictions(self.model, inputs, self.args)
                predicted = (F.softmax(outputs,1).data.argmax(1))
                
                if self.args.log_train_and_meta_metrics:
                    test_loss +=F.cross_entropy(outputs, labels).item()
                
                f1(predicted,labels) #open bug in torchmetrics F1, therefore, softmax must be applied prior to calc
                mcc(outputs,labels)
                
        epoch_f1 = f1.compute()
        epoch_mcc = mcc.compute()
        
        test_loss /= len(test_loader.dataset)
        
        if self.args.log_train_and_meta_metrics:
            tensorboard_writer.add_scalar('val/ Loss',test_loss,epoch)
            tensorboard_writer.add_scalar('val/ F1', epoch_f1, epoch)
            tensorboard_writer.add_scalar('val/ MCC', epoch_mcc, epoch)
            tensorboard_writer.flush()
        
        print(f'Val set: Average loss: {test_loss}')        
        print(f"Val set Avg MCC: {epoch_mcc}")
        
        return epoch_f1, epoch_mcc

    def eval_training_batch(self, f1, mcc, minibatch_loss, epoch, batch_idx, outputs, labels, length_train_loader, tensorboard_writer):

        with torch.no_grad():            
            #-------------        Record Train  Metrics -----------------
            if self.args.log_train_and_meta_metrics:
                train_loss = minibatch_loss.item()
                tensorboard_writer.add_scalar('train/ Loss', train_loss, epoch * length_train_loader + batch_idx)
            preds = (F.softmax(outputs,1).data.argmax(1))
          
            batch_f1_mtr = f1(preds,labels) #open bug in torchmetrics F1, therefore, softmax must be applied prior to calc
            batch_mcc_mtr = mcc(outputs,labels)
            
            if self.args.log_train_and_meta_metrics:
                tensorboard_writer.flush()
        return batch_f1_mtr, batch_mcc_mtr

    def eval_meta_batch(self, f1, mcc, minibatch_loss, epoch, batch_idx, outputs, labels, length_train_loader, tensorboard_writer):

        with torch.no_grad():            
            #-------------        Record Meta  Metrics -----------------
            if self.args.log_train_and_meta_metrics:
                train_loss = minibatch_loss.item()
                tensorboard_writer.add_scalar('meta/ Loss', train_loss, epoch * length_train_loader + batch_idx)
            preds = (F.softmax(outputs,1).data.argmax(1))
          
            batch_f1_mtr = f1(preds,labels) #open bug in torchmetrics F1, therefore, softmax must be applied prior to calc
            batch_mcc_mtr = mcc(outputs,labels)

            if self.args.log_train_and_meta_metrics:
                tensorboard_writer.flush()
        return batch_f1_mtr, batch_mcc_mtr