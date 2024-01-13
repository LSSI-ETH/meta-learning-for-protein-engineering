# 5A12 VEGF and ANG-2 Binding Classification Data

### Meta data
1. The full 5A12 Fab VH and VL sequences along with ANG2 and VEGF are located in `5A12_ANG2_rcsb_pdb_4ZFG.fasta` and `5A12_VEGF_rcsb_pdb_4ZFG.fasta`, from their respective pdb files.
2. The yeast dispaly scFv plasmid along with degenerate oligos used to make library modifying CDRL1 and CDRH2 can be found in the `oligonucleotides` folder on the main page
3. Further information on 5A12 can be found at https://doi.org/10.1074/jbc.M115.662783


### Data generation & file nomenclature
This folder contains the 5A12 scFv binding data for two antigens: VEGF and ANG-2. The raw data files with 'sort' in the title are preprocessed from the NGS runs of different FACS screening rounds of the yeast display 5A12 scFv library for binding to the respective antigens. The sequences were experimentally binned into High, Low/Non-Binding to antigen. 

### Data format
The sequences in the sort files are formatted by concatenating 9 residues from the CDRH2 with 8 residues from the CDRL1, e.g. [CDRH2, CDRL1]. For example,
```
AASeq
TPAGGYTYYFLSSFGVA
TPAGGYTYYFLSRFGVA
TLMGGITVYFLSSFGVA
```
__For reference__: the fist sequence in the above list correspondds to the wild type 5A12 H2 and L1 regions

### Learning tasks
The learning two learning tasks associated with the 5A12 data are Positive and Unlabeled Learning (PUL) and Multi-Antigen Classification from Largely Single-Antigen Data and correspond to figures 5-8 in the paper.

#### 5A12 PUL:
Raw and preprocessed NGS data includes a clean dataset from the final FACS rounds and a noisy dataset consisting of the unsorted library. Note the unsorted library contains a mix both High and Low/Non-Binding sequences.

#### 5A12 Multi-Antigen Classification from Largely Single-Antigen Data:
Raw and preprocessed NGS data includes final FACS rounds of 5A12 screened for binding to either VEGF or ANG-2.


### TODO
1. Add Full VH & VL sequences to sort data files

2. To facilitate ease of use, datasets prepared for machine learning are planned for upload. In the meantime, to reproduce the paper results the preprocessing pipeline can be executed as described on the main page of this repo.
