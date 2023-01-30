#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from train_routines.basemodel import *

#Adapted From: https://github.com/uber-research/learning-to-reweight-examples
class L2RW(BaseModel):
    def __init__(self, args, device):
            super().__init__(args, device)
                
    def train_step(self, train_loader,meta_loader,epoch, tensorboard_writer, batch_size):
        #Higher verison adapted from:
        #https://github.com/TinfoilHat0/Learning-to-Reweight-Examples-for-Robust-Deep-Learning-with-PyTorch-Higher
        print('Epoch: %d' % epoch)
        f1 = F1(num_classes = self.num_classes).to(self.device)
        mcc = MatthewsCorrcoef(num_classes = self.num_classes).to(self.device)
        f1_meta = F1(num_classes = self.num_classes).to(self.device)
        mcc_meta = MatthewsCorrcoef(num_classes = self.num_classes).to(self.device)
        
        self.model.train()
        length_train_loader = len(train_loader.dataset)
        length_meta_loader = len(meta_loader.dataset)
        meta_loader = itertools.cycle(meta_loader)
        
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            
            with higher.innerloop_ctx(self.model, self.optimizer) as (meta_model, meta_opt):
                # 1. Update meta model on training data
                meta_train_outputs = self.get_predictions(meta_model,inputs)
                
                meta_train_loss = F.cross_entropy(meta_train_outputs, labels, reduction = 'none')
                eps = torch.zeros(meta_train_loss.size(), requires_grad=True).to(self.device)
                

                meta_train_loss = torch.sum(eps * meta_train_loss)
                meta_opt.step(meta_train_loss)
    
                # 2. Compute grads of eps on meta validation data
                meta_inputs, meta_labels =  next(meta_loader)
                meta_inputs, meta_labels = meta_inputs.to(self.device), meta_labels.to(self.device)
                
                meta_val_outputs = self.get_predictions(meta_model, meta_inputs)
                
                #criterion.reduction = 'mean'
                meta_val_loss = F.cross_entropy(meta_val_outputs, meta_labels, reduction = 'mean')
                eps_grads = torch.autograd.grad(meta_val_loss, eps)[0].detach()
                
            # 3. Compute weights for current training batch
            w_tilde = torch.clamp(-eps_grads, min=0)
            l1_norm = torch.sum(w_tilde)
            if l1_norm != 0:
                w = w_tilde / l1_norm
            else:
                w = w_tilde
            
            # 4. Train model on weighted batch
            outputs = self.get_predictions(self.model, inputs)
            
            minibatch_loss = F.cross_entropy(outputs, labels, reduction = 'none')
            minibatch_loss = torch.sum(w * minibatch_loss)
            minibatch_loss.backward()
            self.optimizer.step()
            
            batch_f1_mtr,batch_mcc_mtr = super().eval_training_batch(f1, mcc, minibatch_loss, 
                                                                    epoch, batch_idx, outputs, labels, length_train_loader, tensorboard_writer)
            
            meta_f1_mtr,meta_mcc_mtr = super().eval_meta_batch(f1_meta, mcc_meta, meta_val_loss, 
                                                                    epoch, batch_idx, meta_val_outputs, meta_labels, length_meta_loader, tensorboard_writer)

        if self.scheduler is not None:
                self.scheduler.step()

        with torch.no_grad():        
            #epoch_acc = acc.compute()
            epoch_f1 = f1.compute()
            epoch_mcc = mcc.compute()
            epoch_f1_meta = f1_meta.compute()
            epoch_mcc_meta = mcc_meta.compute()
            #tensorboard_writer.add_scalar('train/ Accuracy', epoch_acc, epoch)
            tensorboard_writer.add_scalar('train/ F1', epoch_f1, epoch)
            tensorboard_writer.add_scalar('train/ MCC', epoch_mcc, epoch)
            tensorboard_writer.add_scalar('meta/ F1', epoch_f1_meta, epoch)
            tensorboard_writer.add_scalar('meta/ MCC', epoch_mcc_meta, epoch)
            tensorboard_writer.flush()