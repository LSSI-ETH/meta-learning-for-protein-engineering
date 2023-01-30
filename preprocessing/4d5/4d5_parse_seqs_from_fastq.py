#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from Bio.Seq import Seq #used for nucleotide translation
from Bio.Alphabet import IUPAC #used for nucleotide translation
import ANT #package for handling degenerate codons
import numpy as np
import itertools
import collections
import argparse

'''
This script serves to:
    1) read DNA sequences from fasta file
    2) determine nucleotide sequences possible from a library of degenerate codons
    3) filter fasta dna sequences for presence in degenerate library
    4) translate dna sequences to proteins
    4) output protein sequences to csv file
'''

parser = argparse.ArgumentParser(description='Parse Herceptin scFv sequences from BBDUk fastq files')
parser.add_argument('--infile', type=str, default="input.fastq")
parser.add_argument('--outfile', type=str, default="outfile.csv")

parser.set_defaults(augment=True)
args = parser.parse_args()


def possible_library_sequences(degenrate_codon_list):
    '''
    input:
        degenrate_codon_list: list of possible amplicons in library, wild type & degenerate codon containing strings
           must be divisible by 3. all sequences in list must be the same length

    output:
        output_dict: dictionary containing possible cocons at each position in sequence
    '''
    #collect nucleotide 3mers from degenerate input string
    kmer_list = []
    if np.shape(degenrate_codon_list)[0] > 1:
        for degenrate_codon in degenrate_codon_list:
                kmer_list.append([degenrate_codon[i:i+3] for i in range(0,len(degenrate_codon),3)])
    else:
        kmer_list.append([degenrate_codon[i:i+3] for i in range(0,len(degenrate_codon),3)])
    dict_list = []

    for i in range(len(kmer_list)):
        kmer_dict = {j: ANT.DegenerateCodon(input= kmer_list[i][j] ,table=1).getCodons() for j in range(len(kmer_list[i]))}
        dict_list.append(kmer_dict)

    output_dict = collections.defaultdict(list)
    # Note: the # of entries in this chain should be equal to the length of degenerate_codon_lists
    #       this chain should be updated for each new library this script is applied to

    for key, val in itertools.chain(dict_list[0].items(),dict_list[1].items(),
                                    dict_list[2].items(),dict_list[3].items(),
                                    dict_list[4].items(),dict_list[5].items(),
                                    dict_list[6].items(),dict_list[7].items(),
                                    dict_list[8].items(),dict_list[9].items(),
                                    dict_list[10].items()):
            output_dict[key] += val

    print('Completed Assembly of Possible Library Sequences')
    return output_dict

def check_if_seq_is_in_library(seq_list,library_dict):
    '''
    input:
        seq_list: list of dna sequences
        library_dict: dictionary with keys position of 3mer and values = to possible codons at that position

    output:
        list_of_seqs_in_list_and_library: list of dna sequences present in both sequence list and library_dict
        seqs_not_in_library: list of dna sequences present in NGS file, but not in library_dict
    '''
    list_of_seqs_in_list_and_library = []
    seqs_not_in_library = []
    for i in range(len(seq_list)):
        #break dna_seq to 3mer
        tmp_3mer = [seq_list[i][j:j+3] for j in range(0,len(seq_list[i]),3)]

        #verify 3mers are contained in library, if yes, store sequence to list
        check_nucs_in_library = [tmp_3mer[k] in library_dict[k] for k in range(len(tmp_3mer))]
        if all(check_nucs_in_library):
            list_of_seqs_in_list_and_library.append(seq_list[i])
        else:
            seqs_not_in_library.append(seq_list[i])

    print('Completed Applying Possible Library Sequence Filter to NGS File Amplicons')
    return list_of_seqs_in_list_and_library,seqs_not_in_library

#============================= Load I/O Strings ===================================
print('\nNow Processing BBDuk Ouptut via Python Script')
#Read Fasta File & Save DNA Sequence to List
infile = open(args.infile,'r')
aa_outfile = args.outfile

#===================== Process fastq File ==========================================
#read DNA sequence from FASTA file and save
line_counter  = 1 #used to separate quality scores from
tmp_reads = []
paired_reads = []
#================ Pair Reads & Concatenate ===========================#
print('Now Pairing Interleaved Reads...')
for line in infile:
    if line[0] != '>' and line[0] != '+'  and not line.startswith('@M'): #ensures line title and '+' breaks are skipped
        if line_counter %2 != 0:                #serves to skip read quality scoring line
            tmp_reads.append(line[:-1])
            if len(tmp_reads) == 2:
                paired_reads.append([tmp_reads[0], tmp_reads[1]])
                tmp_reads = []
        line_counter+=1
print('Pairing Complete')
#================ Parse CDRH3 & CDRL3 from Paired Reads ===========================#
R1_parse_string = 'GCAACTTACTACTGT' #in frame 5' primer, 5bp upstream of CDRL3
R2_parse_string = 'GGCCCC' #note this is the reverse complement ending 3bp 3' of final CDRH residue

L3_seq_from_R1 = []
H3_seq_from_R2 = []
start_CDRL = 18
stop_CDRL = 42
start_CDRH = 13
stop_CDRH = 40

for read_pair in paired_reads:
    tmp_r1 = read_pair[0]
    tmp_r2 = read_pair[1]
    r1_start = tmp_r1.find(R1_parse_string)  #position of r1 start parse sequence
    r2_start = tmp_r2.find(R2_parse_string) #position of r2 start parse sequence
    L3_seq_from_R1.append(tmp_r1[r1_start + start_CDRL : r1_start + stop_CDRL ])    #parse L3 from R1
    H3_seq_from_R2.append(tmp_r2[r2_start + start_CDRH: r2_start + stop_CDRH ]) #parse H3 from R2


#===================== Drop DNA Singlets & Incorrect Length CDRs =======================
#============== Save Unique (duplicate) Sequences  for further processing ===================
#Note that at this point in the script, the H3 sequence is the reverse complement
#concatenate CDRs
conc_cdrs = []
for i in range(len(H3_seq_from_R2)):
        conc_cdrs.append(H3_seq_from_R2[i] + L3_seq_from_R1[i])
seq_df = pd.DataFrame()
seq_df['H3revComp + L3'] = conc_cdrs
seq_df['H3revComp'] = H3_seq_from_R2
seq_df['L3'] = L3_seq_from_R1
vc = seq_df['H3revComp + L3'].value_counts()
#eliminate H3 & L3 lists to reduce stored information
H3_seq_from_R2 = []
L3_seq_from_R1 = []
seq_df = seq_df[seq_df['H3revComp + L3'].isin(vc.index[vc.gt( 1 ) ] ) ]
correct_combined_cdr_length = 51
seq_df = seq_df[seq_df['H3revComp + L3'].str.len() == correct_combined_cdr_length]
seq_df = seq_df.drop_duplicates()
#Dropping Sequences with Ambiguous Bases
seq_df = seq_df[~seq_df['H3revComp + L3'].str.contains('N')]

#=============== Convert H3 from Reverse Complement to Standard =======================
h3_rev_comp = [Seq(entry,IUPAC.unambiguous_dna) for entry in seq_df['H3revComp']]
h3_rev_comp = [str(seq.reverse_complement()) for seq in h3_rev_comp]
seq_df['H3'] = h3_rev_comp
concatenated_cdrs_h3_fwd_l3 = []
for i in range(len(h3_rev_comp)):
        concatenated_cdrs_h3_fwd_l3.append(h3_rev_comp[i] + seq_df['L3'].iloc[i])
seq_df['H3 + L3'] = concatenated_cdrs_h3_fwd_l3

#===================== Apply Degenerate Library Sequence Filter ========================

#Ensure a given DNA sequence present in the retained sequence list is possible given the
#degenerate codon sequences used to synthesize and transform the library
h3_wt = 'TGGGGAGGCGACGGCTTCTACGCCATG'
l3_wt = 'CAGCACTACACCACCCCTCCCACG'
wt_h3_l3 = h3_wt + l3_wt
h3_1 = 'TGGNNKNNYNNYGGCTTCTACSHGATG'
h3_2 = 'TGGNNKNNYNNYGGCTTCTACATGATG'
l3_1 = 'CAGTTCTWCNNKTYYCCTCCCGGT'
l3_2 = 'CAGTTCTWCNNKTYYCCTCCCAMC'
l3_3 = 'CAGTTCTWCNNKAYGCCTCCCGGT'
l3_4 = 'CAGTTCTWCNNKAYGCCTCCCAMC'
l3_5 = 'CAGYAYTWCNNKTYYCCTCCCGGT'
l3_6 = 'CAGYAYTWCNNKTYYCCTCCCAMC'
l3_7 = 'CAGYAYTWCNNKAYGCCTCCCGGT'
l3_8 = 'CAGYAYTWCNNKAYGCCTCCCAMC'

#Make L3 degenerate sequences the same as the total concatenated H/L length
l3_1 = h3_wt + l3_1
l3_2 = h3_wt + l3_2
l3_3= h3_wt + l3_3
l3_4 = h3_wt + l3_4
l3_5 = h3_wt + l3_5
l3_6 = h3_wt + l3_6
l3_7= h3_wt + l3_7
l3_8 = h3_wt + l3_8
degen_codon_list = [wt_h3_l3, h3_1, h3_2, l3_1, l3_2, l3_3, l3_4, l3_5, l3_6, l3_7, l3_8]

print('Generating Possible Sequences Contained in Degenerate Codons...')
lib_seq_dict = possible_library_sequences(degen_codon_list)
print('Filtering Library Sequences by Degenerate Codons...')
seqs_in_library, out_lib_seqs= check_if_seq_is_in_library(concatenated_cdrs_h3_fwd_l3, lib_seq_dict)

#=============== Assemble Full DNA Sequences ======================================
dna_output_df = pd.DataFrame()
h3_in_lib = [h3_sequence[0:27] for h3_sequence in seqs_in_library]
l3_in_lib = [l3_sequence[27:] for l3_sequence in seqs_in_library]
dna_output_df['H3'] = h3_in_lib
dna_output_df['L3'] = l3_in_lib
dna_output_df['H3 +L3'] = seqs_in_library

#=============== Translate Sequences =============================================
#Convert Retained Sequences to Biopython Format For Translation
dna_biopython = [Seq(entry,IUPAC.unambiguous_dna) for entry in dna_output_df['H3 +L3']]

#Translating 
peptides = [entry.translate(to_stop=True) for entry in dna_biopython]
peptides_series = pd.Series(str(entry) for entry in peptides)

#Removing Sequences Containing Premature Stop Codons
peptide_length = 17
peptides_filtered_by_length = peptides_series[peptides_series.str.len() == peptide_length]
unique_filtered_peptides = pd.Series(peptides_filtered_by_length.unique(),name='AASeq')
unique_filtered_peptides.to_csv(aa_outfile, header=True,index=False)