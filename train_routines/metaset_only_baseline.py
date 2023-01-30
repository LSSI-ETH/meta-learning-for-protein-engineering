#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from train_routines.basemodel import *

class MetaSetOnlyBaseline(BaseModel):
    def __init__(self, args, device):
            super().__init__(args, device)
        
    def train_step(self,train_dataloader,meta_loader, epoch, tensorboard_writer, batch_size):
        train_loss = 0

        f1 = F1(num_classes = self.num_classes).to(self.device)
        mcc = MatthewsCorrcoef(num_classes = self.num_classes).to(self.device)
        
        self.model.train()
        
        for batch, (X, labels) in enumerate(meta_loader):
            X = X.to(self.device)
            labels = labels.to(self.device)
            pred = self.get_predictions(self.model, X)                
            loss = self.loss_fn(pred,labels)
            train_loss = loss.item()
            tensorboard_writer.add_scalar('train/ Loss', train_loss, epoch * len(meta_loader) + batch)
            
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

        print('\nTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch, len(meta_loader),
                        100. * batch / len(meta_loader), train_loss))
        print(f'Train F1 score: {epoch_f1 }')
        print(f'Train MCC: {epoch_mcc}')
        
        tensorboard_writer.add_scalar('train/ F1', epoch_f1, epoch)
        tensorboard_writer.add_scalar('train/ MCC', epoch_mcc, epoch)
        tensorboard_writer.flush()