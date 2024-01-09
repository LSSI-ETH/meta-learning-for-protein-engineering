#!/bin/bash

#path to bbduk folder
bbduk_path="/Users/user/Documents/BBTools/bbmap/"

#path to raw deep sequencing data
datapath="https://github.com/LSSI-ETH/meta-learning-for-protein-engineering/data/raw_ngs/4d5/"

#path to preprocessing scripts
preprocessing_path="https://github.com/LSSI-ETH/meta-learning-for-protein-engineering/preprocessing/4d5/"

#path to save preprocessed data
outputpath="https://github.com/LSSI-ETH/meta-learning-for-protein-engineering/data/4d5/"

cd $bbduk_path

arr_r=("4d5_neg_initial_sort" "4d5_neg_4th_sort" \
"4d5_low_initial_sort" "4d5_low_4th_sort" \
"4d5_high_initial_sort" "4d5_high_4th_sort")

for i in "${!arr_r[@]}"; do
    reads1="$datapath${arr_r[$i]}_R1.fastq.gz"
    reads2="$datapath${arr_r[$i]}_R2.fastq.gz"
    clean1="${datapath}clean1.fq"
    clean2="${datapath}clean2.fq"
    
    #quality trimming
    ./bbduk.sh in1=$reads1 in2=$reads2 out1=$clean1 out2=$clean2 qtrim=r trimq=22

    bbdoutput="$datapath${arr_r[$i]}.fq"
    #echo "$outpath"
    #interleave reads
    ./reformat.sh in1=$clean1 in2=$clean2 out=$bbdoutput
    
    cd $preprocessing_path
    python 4d5_parse_seqs_from_fastq.py --infile=$bbdoutput --outfile="${outputpath}${arr_r[$i]}.csv"
    #remove leftover files
    rm $clean1 $clean2 $bbdoutput

    cd $bbduk_path
  done

cd $preprocessing_path

python 4d5_syn_train_test_split.py

python 4d5_syn_truncate.py

python 4d5_exp_train_test_split.py

python 4d5_exp_truncate.py