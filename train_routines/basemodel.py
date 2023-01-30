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
from torchmetrics import Accuracy, F1, MatthewsCorrcoef

class BaseModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()

        self.device = device
        
        if args.data_type == '4d5_syn' or args.data_type == '4d5_exp': self.num_classes = 3
        elif args.data_type == '5a12_2ag': self.num_classes = 4
        elif args.data_type.startswith('5a12_PUL'): self.num_classes = 2
        
        self.input_size = 17 #input sequence length
        self.ntokens = 21 #transformer embedding vocabulary

        #base model selection
        if args.base_model == 'cnn':
            filters, dense  = 512, 512
            self.model = CNN(input_size = self.input_size, conv_filters = filters, 
                                 dense_nodes = dense, n_out = self.num_classes, kernel_size = args.kernel, dropout = args.dropout).to(self.device)

        elif args.base_model == 'transformer' :
            self.model = Transformer(ntoken = self.ntokens, emb_dim = 32, nhead = 2, nhid = 128, nlayers = 1, 
                     n_classes = self.num_classes, seq_len = self.input_size, dropout = args.dropout, 
                     out_dim = 512).to(self.device)

        elif args.base_model == 'logistic_regression' :
            self.model = LogisticRegression(input_size = self.input_size, n_classes = self.num_classes).to(self.device).to(self.device)
        
        elif args.base_model == 'mlp':
            self.model = MLP(input_size = self.input_size, hdim1 = 256, hdim2 = 512, 
                             n_out = self.num_classes, dropout = args.dropout).to(self.device)

        #learning rate, & lr schedule, loss fn
        self.learning_rate = args.learn_rate

        if args.opt_id == 'sgd': self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        elif args.opt_id == 'adam':  self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, amsgrad=True, eps = 1e-8)
        
        else: self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
            
        self.lr_scheduler = args.lr_scheduler
        
        if self.lr_scheduler: self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[15,30], gamma=0.1)
        else: self.scheduler = None
        
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)

    def get_predictions(self,model, X, mask = None):

        if model.__class__.__name__ =='Transformer' or model.__class__.__name__ =='FunctionalTransformer':        
            pred = model(X)
        else:       
            pred = model(X.float())
        return pred

    def mlc_get_predictions(self,model, X):
        
        if model.__class__.__name__ =='Transformer' or model.__class__.__name__ =='FunctionalTransformer':        
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
            X = X.to(self.device)
            labels = labels.to(self.device)
            pred = self.get_predictions(self.model, X)                
            loss = self.loss_fn(pred,labels)
            train_loss = loss.item()
            tensorboard_writer.add_scalar('train/ Loss', train_loss, epoch * len(train_dataloader) + batch)
            
            self.optimizer.zero_grad() 
            loss.backward()
            self.optimizer.step()
            
            if self.scheduler is not None: self.scheduler.step()            
        
            with torch.no_grad():
                predicted = (F.softmax(pred,1).data.argmax(1))
                batch_f1_mtr = f1(predicted,labels) #open bug in torchmetrics F1, therefore, softmax must be applied prior to calc
                batch_mcc_mtr = mcc(pred,labels)
                
        epoch_f1 = f1.compute()
        epoch_mcc = mcc.compute()

        print('\nTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch, len(train_dataloader),
                        100. * batch / len(train_dataloader), train_loss))
        print(f'Train F1 score: {epoch_f1 }')
        print(f'Train MCC: {epoch_mcc}')
        
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
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.get_predictions(self.model, inputs)
                test_loss +=F.cross_entropy(outputs, labels).item()
                predicted = (F.softmax(outputs,1).data.argmax(1))
                
                f1(predicted,labels) #open bug in torchmetrics F1, therefore, softmax must be applied prior to calc
                mcc(outputs,labels)
                
        epoch_f1 = f1.compute()
        epoch_mcc = mcc.compute()
        test_loss /= len(test_loader.dataset)
        
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
            train_loss = minibatch_loss.item()
            tensorboard_writer.add_scalar('train/ Loss', train_loss, epoch * length_train_loader + batch_idx)
            preds = (F.softmax(outputs,1).data.argmax(1))
          
            batch_f1_mtr = f1(preds,labels) #open bug in torchmetrics F1, therefore, softmax must be applied prior to calc
            batch_mcc_mtr = mcc(outputs,labels)
            
            if batch_idx % 10000 == 0:
                print('\nEpoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format( epoch, batch_idx, length_train_loader, 100. * batch_idx / length_train_loader, train_loss))
                print(f'Train F1 score: {batch_f1_mtr}')
                print(f'Train MCC score: {batch_mcc_mtr}')
            tensorboard_writer.flush()
        return batch_f1_mtr, batch_mcc_mtr

    def eval_meta_batch(self, f1, mcc, minibatch_loss, epoch, batch_idx, outputs, labels, length_train_loader, tensorboard_writer):

        with torch.no_grad():            
            #-------------        Record Train  Metrics -----------------
            train_loss = minibatch_loss.item()
            tensorboard_writer.add_scalar('meta/ Loss', train_loss, epoch * length_train_loader + batch_idx)
            preds = (F.softmax(outputs,1).data.argmax(1))
          
            batch_f1_mtr = f1(preds,labels) #open bug in torchmetrics F1, therefore, softmax must be applied prior to calc
            batch_mcc_mtr = mcc(outputs,labels)
            
            if batch_idx % 10000 == 0:
                print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format( epoch, batch_idx, length_train_loader, 100. * batch_idx / length_train_loader, train_loss))
                print(f'Meta F1 score: {batch_f1_mtr}')
                print(f'Meta MCC score: {batch_mcc_mtr}')
            tensorboard_writer.flush()
        return batch_f1_mtr, batch_mcc_mtr

