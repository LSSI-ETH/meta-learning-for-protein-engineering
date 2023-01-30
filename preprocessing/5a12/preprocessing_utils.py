#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from Levenshtein import distance as levenshtein_distance
import pandas as pd
import numpy as np
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt

def add_LD_to_df(data_frame, seq_col_str, consensus_sequence):
    """
    calculates edit distance from consensus sequence for each amino acid sequence in dataframe. returns 
    dataframe with appended LD column

    Parameters
    ----------
    data_frame : pandas.Dataframe
        dataframe containing amino acid sequences
    seq_col_str : str
        string of column ID containing sequences in dataframe    
    consensus_sequence : str
        string containing consensus sequence of full data set.

    Returns
    -------
    data_frame_out : pandas.DataFrame
        data_frame with the column LD appended to it, containing the edit distance for each sequence from the consensus sequence.

    """

    wt_str = consensus_sequence
    
    LD_arr = []
    for i in range(len(data_frame)):
        LD_arr.append( levenshtein_distance(wt_str, data_frame[seq_col_str].iloc[i]) )
    data_frame_out = data_frame.copy()
    data_frame_out['LD'] = LD_arr
    
    return data_frame_out

def consensus_seq_and_LD_to_df(data_frame, seq_col_str = 'aaseq', return_df = True):
    
    #============================================================================== 
    #======= Calculate residue-position frequencies & store in seq_count_df =======
    #============================================================================== 
    sequences = pd.Series(data_frame[seq_col_str]).to_list()
       
    max_len = max(map(len, sequences))
    seq_count_dict = defaultdict(lambda: [0]*max_len)  # d[char] = [pos0, pos12, ...]
    for seq in sequences:
        for i, char in enumerate(seq): 
            seq_count_dict[char][i] += 1
       
    seq_count_df = pd.DataFrame.from_dict(seq_count_dict)
    seq_count_df = seq_count_df.T
    seq_count_df.columns = [str(i) for i in range(len(sequences[0]))]
    consensus_seq = list(seq_count_df.idxmax(axis=0))
    consensus_seq = ''.join(consensus_seq)
    #============================================================================== 
    #============================================================================== 
    data_frame = add_LD_to_df(data_frame, seq_col_str = seq_col_str, consensus_sequence = consensus_seq)
    
    if return_df == True:    return data_frame, consensus_seq
    else: return consensus_seq
    
def combine_df_list_and_shuffle(df_list, keep = False):
    '''
    combines two dataframes, drops duplicates, & shuffles
    
    Parameters
    ----------

    data_frame : pandas.DataFrame
        dataframe containing all sequence & label data
    keep: bool
        whether or not to keep duplicates

    Returns
    -------
    data_frame : pandas.DataFrame
        combined, shuffled dataframe
    '''
    frames = df_list
    common_cols = list(set.intersection(*(set(df.columns) for df in frames)))
    combined_df = pd.concat([df[common_cols] for df in frames], ignore_index=True).drop_duplicates(subset='AASeq', keep=keep)
    combined_df = combined_df.sample(frac = 1, random_state = 1)
    return combined_df
