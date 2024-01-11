# 4D5 Trastuzumab Binding Classification Data

### Meta data
1. The full 4D5 Fab VH and VL sequences along with HER2 are located in `Herceptin_Fab_HER2_rcsb_pdb_1N8Z.fasta`, from pdb `1N8Z`
2. The yeast dispaly scFv plasmid along with degenerate oligos used to make library modifying CDRL1 and CDRH2 can be found in the `oligonucleotides` folder on the main page
3. Further information on 4D5 can be found at https://www.ncbi.nlm.nih.gov/pmc/articles/PMC49066/

### Data generation & file nomenclature
This folder contains the 4D5 scFv trastuzumab binding data. The raw data files with 'sort' in the title are preprocessed from the NGS runs of different FACS screening rounds of the yeast display 4D5 scFv library for binding to trastuzumab. The sequences were experimentally binned into High, Low, and Non-Binding to trastuzumab. 

### Data format
The sequences in the sort files are formatted by concatenating 9 residues from the CDRH3 with 8 residues from the CDRL3, e.g. [CDRH3, CDRL3]. For example,
```
AASeq
WGGDGFYAMQHYTTPPT
WNHDGFYQMQFYDSPPG
WWSPGFYMMQFFCSPPN
```
where the fist sequence in the above list correspondds to the wild type 4D5 H3 and L3 regions


### Learning tasks
The learning tasks associated with the 4D5 data include a clean dataset from the final FACS rounds and a noisy dataset utilizing the first FACS round for train and the final FACS round for meta and test sets and corresponds to figures 3 and 4 in the paper.


### TODO
1. Add Full VH & VL sequences to sort data files

2. To facilitate ease of use, datasets prepared for machine learning are planned for upload. In the meantime, to reproduce the paper results the preprocessing pipeline can be executed as described on the main page of this repo.