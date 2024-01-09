#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from train_routines.basemodel import *

class MetaSetOnlyBaseline(BaseModel):
    def __init__(self, args, device):
            super().__init__(args, device)
            self.args = args
        
    def train_step(self,train_dataloader,meta_loader, epoch, tensorboard_writer, batch_size):
        

        f1 = F1(num_classes = self.num_classes).to(self.device)
        mcc = MatthewsCorrcoef(num_classes = self.num_classes).to(self.device)
        
        self.model.train()
        
        for batch, (X, labels) in enumerate(meta_loader):
            X = X.to(self.device)
            labels = labels.to(self.device)
            pred = self.get_predictions(self.model, X, self.args)                
            loss = self.loss_fn(pred,labels)
            
            self.optimizer.zero_grad() 
            loss.backward()
            self.optimizer.step()
            
            if self.scheduler is not None: self.scheduler.step()            
        
            with torch.no_grad():
                predicted = (F.softmax(pred,1).data.argmax(1))
                batch_f1_mtr = f1(predicted,labels) #note as of 2022 there is an open bug in torchmetrics F1, therefore, softmax must be applied prior to calc
                batch_mcc_mtr = mcc(pred,labels)
                
        epoch_f1 = f1.compute()
        epoch_mcc = mcc.compute()