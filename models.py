#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np




#================== Transformer ==========================
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # shape [1, seq_len, emb_dim]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :] #shape [batch_size, seq_len, emb_dim]
        return self.dropout(x)       
        
class Transformer(nn.Module):

    def __init__(self, ntoken, emb_dim, nhead, nhid, nlayers, n_classes, 
                 seq_len, dropout=0.2, out_dim = 512):
        super(Transformer, self).__init__()
        self.pos_encoder = PositionalEncoding(emb_dim, dropout)
        encoder_layers = TransformerEncoderLayer(emb_dim, nhead, nhid, 
                                                 dropout, 
                                                 batch_first = True,
                                                 norm_first = True, 
                                                 activation = 'gelu')
        
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, emb_dim)
        self.emb_dim = emb_dim
        self.relu = nn.GELU()
        self.seq_len = seq_len
        self.dropout = nn.Dropout(p=dropout)        
        self.flatten = nn.Flatten()
        self.decoder = nn.Linear( int(seq_len * emb_dim), out_dim)
        self.out_layer = nn.Linear( out_dim , n_classes )
        
        
        self.transformer_encoder.apply(self.init_weights)
        self.encoder.apply(self.init_weights)
        self.decoder.apply(self.init_weights)
        self.out_layer.apply(self.init_weights)
        
    def init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            stdv = 1. / math.sqrt(module.weight.size(1))
            module.weight.data.uniform_(-stdv, stdv)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.uniform_(-stdv, stdv)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def forward(self, src, input_mask = None, return_h = False):
        src = self.encoder(src) * math.sqrt(self.emb_dim)
        src = self.pos_encoder(src)
        if input_mask is not None:
            output = self.transformer_encoder(src, src_key_padding_mask = input_mask)
        elif input_mask == None:
            output = self.transformer_encoder(src)
        output = self.flatten(output) 
        output = self.decoder(output)
        output = self.relu(output)
        output = self.dropout(output)
        
        if return_h: return self.out_layer(output), output
        else: return self.out_layer(output)


#================== CNN ==========================
class CNN(nn.Module):
    def __init__(self, input_size, conv_filters, dense_nodes,n_out, 
                 kernel_size, dropout):
        super(CNN, self).__init__()
        
        pad, dilation, stride_var = 0, 1, 1 #note original padding is 1
        maxpool_kernel = 2
        kernel_2 = 3
        mp_stride = maxpool_kernel
        mp_pad = 1
        input_vector_len = 21
        
        conv_out_size  = math.floor( ( ( input_vector_len + 2*pad - dilation*(kernel_size - 1) - 1 ) /stride_var) + 1)
        mp_out_size = math.floor( ( ( conv_out_size + 2*mp_pad - dilation*(maxpool_kernel - 1) - 1 ) /mp_stride) + 1)
        conv_out_size2 = math.floor( ( ( mp_out_size + 2*pad - dilation*(kernel_2 - 1) - 1 ) /stride_var) + 1)
        mp_out_size2 = math.floor( ( ( conv_out_size2 + 2*mp_pad - dilation*(maxpool_kernel - 1) - 1 ) /mp_stride) + 1)
        
        transition_nodes =  math.floor( (conv_filters / 2) * mp_out_size2)
        
        self.conv_bn_relu_stack = nn.Sequential(
            
            nn.Conv1d(input_size, conv_filters, kernel_size = kernel_size,
                      padding=pad, stride = stride_var, bias=False),
            nn.MaxPool1d(kernel_size=maxpool_kernel, padding = mp_pad),
            nn.ReLU(),
            nn.BatchNorm1d(conv_filters),
            

            nn.Conv1d(conv_filters, int(conv_filters / 2), 
                      kernel_size = kernel_2, padding=pad, stride = stride_var, 
                      bias=False),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=maxpool_kernel, padding = mp_pad),
            nn.BatchNorm1d(int(conv_filters/2)),

            )

        self.linear_relu_stack = nn.Sequential(
            
            nn.Linear(transition_nodes, dense_nodes), 
            nn.ReLU(),
            
            )

        self.flatten = nn.Flatten()        
        self.dropout = nn.Dropout(p=dropout)        
        self.out_layer = nn.Linear(dense_nodes,n_out)
        
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x, mask = None, return_h = False):
        x = x.float()
        if mask is not None:
            x = x * mask
        x = self.conv_bn_relu_stack(x)
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        x = self.dropout(x)
        
        if return_h: return self.out_layer(x), x
        else: return self.out_layer(x)



#===================== Logistic Regression ==============
class LogisticRegression(nn.Module):

    def __init__(self, input_size, n_classes):
        super(LogisticRegression, self).__init__()
        
        input_vector_len = 21 #amino acid vocabulaary for one-hot encoding
        input_dim = int( input_size * input_vector_len)
        self.linear = nn.Linear(input_dim, n_classes)
        self.flatten = nn.Flatten()
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, input_mask = None, return_h = False):
        x = self.flatten(x)
        if return_h: return self.linear(x), x
        else: return self.linear(x)

#===================== MultiLayer Perceptron ==============
class MLP(nn.Module):
    def __init__(self, input_size, hdim1, hdim2, n_out, dropout):
        super(MLP, self).__init__()
        
        input_vector_len = 21 #amino acid vocabulaary for one-hot encoding
        self.flatten = nn.Flatten()        
        self.net = nn.Sequential(
            nn.Linear(int(input_size * input_vector_len), hdim1),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hdim1, hdim2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            )
        self.out_layer = nn.Linear(hdim2,n_out)
            
        def init_weights(self):
            
            nn.init.kaiming_uniform_(self.net[0].weight)
            nn.init.kaiming_uniform_(self.net[2].weight)
            nn.init.xavier_normal_(self.net[4].weight)
    
            self.net[0].bias.data.zero_()
            self.net[2].bias.data.zero_()

    def forward(self, x, mask = None, return_h = False):
        if mask is not None:
            x = x * mask
        x = self.flatten(x)
        x = self.net(x)
        
        if return_h: return self.out_layer(x), x
        else: return self.out_layer(x)