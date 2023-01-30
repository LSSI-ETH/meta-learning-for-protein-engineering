#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

def set_learn_rates(args):
    if args.base_model == 'cnn':
        args.learn_rate = 5e-3
        args.vnet_lr = 1e-4
            
    elif args.base_model == 'transformer':
        if args.top_model == 'l2rw': 
            args.learn_rate = 1e-3
        elif args.top_model =='mlc':
            args.vnet_lr = 1e-3
            args.learn_rate = 1e-2
        else:
            args.learn_rate = 5e-3


#===========================   DataSet & Loaders         ================================ 

class CustomTorchDataset(Dataset):
    """
    Converts categorically encoded sequences & labels into a torch Dataset
    
    Parameters
    ----------
    encoded_seqs: list
        categorically encoded protein or nucleotide sequences
    labels: list
        class labels or regression fitness values corresponding to sequences

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
    Collater function to pad sequences of variable length (AAV data) and calculate padding mask. Fed to 
    torch DataLoader collate_fn.
    
    Parameters
    ----------
    alphabet: str
        vocabulary size (i.e. amino acids, nucleotide ngrams). used for one-hot encoding dimension calculation
    pad_tok: float 
        padding token. zero padding is used as default
    args: argparse.ArgumentParser
        arguments specified by user. used for this program to determine one-hot or categorical encoding

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
        
        if self.args.base_model != 'transformer':
            padded = F.one_hot(padded, num_classes = self.vocab_length)
        
        return padded, y

#===========================   Convert Data to torch.DataLoader        ======================

def data_to_loader(x_train,x_val, x_test, x_meta, y_train, y_val, y_test, y_meta, args):
    """
    Function for converting categorically encoding sequences + their labels to a torch Dataset and DataLoader
    
    Parameters
    ----------
    x_train, x_val, x_test: list
        categorically encoded protein or nucleotide training, validation, and testing sequences
    y_train, y_val, y_test: pandas.core.series.Series
        class labels corresponding to training, validation, & testing sequences
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
    
    if args.base_model == 'transformer':
        if len(y_meta) < batch_size : drop_last_meta_bool = False
        else: drop_last_meta_bool = True
        
        if len(y_train) < batch_size : drop_last_train_bool = False
        else: drop_last_train_bool = True
        
        if len(y_val) < batch_size : drop_last_val_bool = False
        else: drop_last_val_bool = True
    else:
        drop_last_meta_bool, drop_last_train_bool, drop_last_val_bool = False, False, False
        
    vocab_length = 21

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, 
                                               collate_fn=Collater(vocab_length = vocab_length, pad_tok=0., args=args), drop_last=drop_last_train_bool)
    meta_loader = torch.utils.data.DataLoader(meta_data, batch_size=batch_size, shuffle=True,
                                              collate_fn=Collater(vocab_length = vocab_length, pad_tok=0., args=args), drop_last=drop_last_meta_bool)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False,
                                              collate_fn=Collater(vocab_length = vocab_length, pad_tok=0., args=args), drop_last=drop_last_val_bool)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                               collate_fn=Collater(vocab_length = vocab_length, pad_tok=0., args=args), drop_last=True)

    return train_loader, val_loader, test_loader, meta_loader


#===========================   Categorically Encode ngrams    ==========================

def encode_ngrams(x,args):
    """
    Converts amino acid or nucleotide sequences to categorically encoded vectors based on a chosen
    encoding approach (ngram vocabulary).    
    
    Parameters
    ----------
    x: pandas.core.series.Series
        pandas Series containing strings of protein or nucleotide training, validation, or testing sequences
    args: argparse.ArgumentParser
        arguments specified by user. used for this program to determine correct vocabulary size, output 
        shape, and if a mask should be returned

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