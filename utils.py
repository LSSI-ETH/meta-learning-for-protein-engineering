#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import os

def set_learn_rates(args):
    if args.base_model == 'cnn':
        args.learn_rate = 5e-3
        args.vnet_lr = 1e-4
            
    elif 'transformer' in args.base_model:
        if args.top_model == 'l2rw': 
            args.learn_rate = 1e-3
        elif args.top_model =='mlc':
            args.vnet_lr = 1e-3
            args.learn_rate = 1e-2
        else:
            args.learn_rate = 5e-3


def save_checkpoint(model, optimizer, epoch, gpu, args):

    print("epoch: {} ".format(epoch+1))
    checkpointing_path = args.checkpoint_path + f'checkpoint_{args.param_file}.pth'
    print("Saving the Checkpoint: {}".format(checkpointing_path))
    torch.save({
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, checkpointing_path)

def load_checkpoint(model, optimizer, gpu, args):
    
    print("--------------------------------------------")
    print("Checkpoint file found!")
    print("Loading Checkpoint From: {}".format(args.checkpoint_path + f'checkpoint_{args.param_file}.pth'))
    
    if torch.cuda.device_count() > 0:
        map_location = torch.device('cuda')
    else:
        map_location = torch.device('cpu')
        
    checkpoint = torch.load(args.checkpoint_path + f'checkpoint_{args.param_file}.pth', map_location=map_location)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_number = checkpoint['epoch']

    print("Checkpoint File Loaded - epoch_number: {}".format(epoch_number))
    print('Resuming training from epoch: {}'.format(epoch_number+1))
    print("--------------------------------------------")
    return model, optimizer, epoch_number



#===========================   DataSet & Loaders         ================================ 

class CustomTorchDataset(Dataset):
    """
    Converts categorically encoded sequences & labels into a torch Dataset
    
    Parameters
    ----------
    encoded_seqs: list
        categorically encoded protein sequences
    labels: list
        class labels 

    Returns
    -------
    tuple of sequences, labels (y)
    """    
    def __init__(self, encoded_seqs, labels, transform=None):
        self.encoded_seqs = encoded_seqs
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.labels) 
    def __getitem__(self, idx):
        seq = self.encoded_seqs[idx]
        label = self.labels[idx]
        if self.transform:
            seq = self.transform(seq)
        return seq, label
    
#===========================   Collater Fn to Apply Padding         ====================

class Collater(object):
    """
    Collater function to pad sequences of variable length (if appropriate) and calculate padding mask. Fed to 
    torch DataLoader collate_fn.
    
    Parameters
    ----------
    alphabet: str
        vocabulary size (i.e. amino acids). used for one-hot encoding dimension calculation
    pad_tok: float 
        padding token. zero padding is used as default
    args: argparse.ArgumentParser

    Returns
    -------
    padded sequences, labels (y)
    """    
    def __init__(self, vocab_length: int, 
                pad_tok=0,
                args = None):        
        self.vocab_length = vocab_length
        self.pad_tok = pad_tok
        self.args = args

    def __call__(self, batch):
        data = tuple(zip(*batch))
        sequences = data[0]        
        y = data[1]
        y = torch.tensor(y).squeeze()
        y = y.type(torch.LongTensor)
        
        maxlen = sequences[0].shape[0]
        padded = torch.stack([torch.cat([i, i.new_zeros(maxlen - i.size(0))], 0) for i in sequences],0)
        
        if 'transformer' not in self.args.base_model:
            padded = F.one_hot(padded, num_classes = self.vocab_length)
        
        return padded, y

#===========================   Convert Data to torch.DataLoader        ======================

def data_to_loader(x_train,x_val, x_test, x_meta, y_train, y_val, y_test, y_meta, args):
    """
    Function for converting categorically encoding sequences + their labels to a torch Dataset and DataLoader
    
    Parameters
    ----------
    x_train, x_val, x_test: list
        categorically encoded protein sequences
    y_train, y_val, y_test: pandas.core.series.Series
        class labels 
    batch_size: int
        batch size to be used for dataloader
    args: argparse.ArgumentParser
        arguments specified by user. 
    alphabet: list of strings
        vocabulary (i.e. amino acids). passed to collate_fn and used for one-hot 
        encoding dimension calculation
    Returns
    -------
    torch DataLoader objects for training, validation, testing, meta sets
    """    
    
    batch_size = args.batch_size
    
    y_train = y_train.to_list()
    y_meta = y_meta.to_list()
    y_val = y_val.to_list()
    y_test = y_test.to_list()


    train_data = CustomTorchDataset(x_train, y_train, transform = None)
    meta_data = CustomTorchDataset(x_meta, y_meta, transform = None)
    val_data = CustomTorchDataset(x_val, y_val, transform = None)
    test_data = CustomTorchDataset(x_test, y_test, transform = None)
    
    if 'transformer' in args.base_model:
        if len(y_meta) < batch_size : drop_last_meta_bool = False
        else: drop_last_meta_bool = True
        
        if len(y_train) < batch_size : drop_last_train_bool = False
        else: drop_last_train_bool = True
        
        if len(y_val) < batch_size : drop_last_val_bool = False
        else: drop_last_val_bool = True

    #necessary for 5a12_2ag edit_distance 6
    elif args.data_type == '5a12_2ag' and args.edit_distance == 6:
        drop_last_meta_bool, drop_last_train_bool, drop_last_val_bool = False, True, False
        
    else:
        drop_last_meta_bool, drop_last_train_bool, drop_last_val_bool = False, False, False
        
    vocab_length = 21
    
    if args.non_block:
        num_works = 2
        pinned_mem = True
    else:
        num_works = 0
        pinned_mem = False
        
    train_loader = torch.utils.data.DataLoader(train_data, 
                                               batch_size=batch_size, 
                                               shuffle=True, 
                                               collate_fn=Collater(vocab_length = vocab_length,
                                                                   pad_tok=0.,
                                                                   args=args), 
                                               drop_last=drop_last_train_bool,
                                               num_workers = num_works,
                                               pin_memory = pinned_mem)
    
    meta_loader = torch.utils.data.DataLoader(meta_data, 
                                              batch_size=batch_size, 
                                              shuffle=True,
                                              collate_fn=Collater(vocab_length = vocab_length,
                                                                  pad_tok=0.,
                                                                  args=args), 
                                              drop_last=drop_last_meta_bool,
                                              num_workers = num_works,
                                              pin_memory = pinned_mem)
    
    val_loader = torch.utils.data.DataLoader(val_data, 
                                             batch_size=batch_size, 
                                             shuffle=False,
                                             collate_fn=Collater(vocab_length = vocab_length,
                                                                 pad_tok=0.,
                                                                 args=args), 
                                             drop_last=drop_last_val_bool,
                                             num_workers = num_works,
                                             pin_memory = pinned_mem)
    
    test_loader = torch.utils.data.DataLoader(test_data, 
                                              batch_size=batch_size, 
                                              shuffle=False,
                                              collate_fn=Collater(vocab_length = vocab_length,
                                                                  pad_tok=0.,
                                                                  args=args), 
                                              drop_last=True,
                                              num_workers = num_works,
                                              pin_memory = pinned_mem)

    return train_loader, val_loader, test_loader, meta_loader


#===========================   Categorically Encode ngrams    ==========================

def encode_ngrams(x,args):
    """
    Converts amino acid to categorically encoded tensors.    
    
    Parameters
    ----------
    x: pandas.core.series.Series
        pandas Series containing strings of proteinsequences

    Returns
    -------
    x_train_idx, x_val_idx, x_test_idx: list
        categorically encoded sequences
    vocabulary:
        vocabulary used for ngram encoding. to be passed to dataloaer & collate functions
    """    
    def seq_to_cat(seq_df, word_to_idx_dictionary):
        '''
        input: dataframe of sequences & dictionary containing tokens in vocabulary
        output: out_idx: list of torch.Tensors of categorically encoded (vocab index) ngrams 
        '''
        out_idxs = []
        
        if isinstance(seq_df,pd.Series): seq_df = seq_df.to_list()
            
        for i in range(len(seq_df)): out_idxs.append(torch.tensor([word_to_idx_dictionary[w] for w in seq_df[i] if w != None and w != '' ], dtype=torch.long))
        
        return out_idxs

    vocabulary = ['UNK', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I','L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']    
    word_to_ix = {word: i for i, word in enumerate(vocabulary)}
    x_idx = seq_to_cat(x, word_to_ix)
    
    return x_idx


#===========================   4D5 Synthetic Noise Methods              ==============================

def flip_labels_C(corruption_prob, num_classes, seed=1):
    '''
    returns a matrix with (1 - corruption_prob) on the diagonals, and corruption_prob
    concentrated in only one other entry for each row
    '''
    np.random.seed(seed)
    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i])] = corruption_prob
    return C

def inject_noise(noise_method, labels, corruption_prob, num_classes, seed=1):
    print(f'Now Injecting Noise Type {str(noise_method)} at Noise Fraction {corruption_prob}')
    C = noise_method(corruption_prob, num_classes, seed)
    corrupted_labels = []
    for i in range(len(labels)):
        corrupted_labels.append(np.random.choice(num_classes, p=C[labels.iloc[i]]))
    return pd.Series(corrupted_labels)


def noise_injection(args, y_train, seed_entry):       
    
    if args.noise_type != 'none': 
        if args.noise_type == 'flip':
            noise_fn = flip_labels_C
        else:
            raise Exception('Uknown noise type ', args.noise_type)
        
        y_train = inject_noise(noise_fn, y_train, args.noise_fraction, args.num_classes, seed_entry)

    return y_train
