#!/bin/bash

#path to bbduk folder
bbduk_path="/Users/mminot/Documents/BBTools/bbmap/"

#path to raw deep sequencing data
datapath="/Users/mminot/Documents/models/meta_2022/meta-learning-protein-engineering/data/raw_ngs/5a12/"

#path to preprocessing scripts
preprocessing_path="/Users/mminot/Documents/models/meta_2022/meta-learning-protein-engineering/preprocessing/5a12/"

#path to save preprocessed data
outputpath="/Users/mminot/Documents/models/meta_2022/meta-learning-protein-engineering/data/5a12/"

cd $bbduk_path

arr_r=("5a12_ang2_neg_2nd_sort" "5a12_ang2_pos_2nd_sort" \
"5a12_vegf_neg_3rd_sort" "5a12_vegf_pos_4th_sort" \
"5a12_vegf_initial_sort") 

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
    python 5a12_parse_seqs_from_fastq.py --infile=$bbdoutput --outfile="${outputpath}${arr_r[$i]}.csv"
    #remove leftover files
    rm $clean1 $clean2 $bbdoutput

    cd $bbduk_path
  done

cd $preprocessing_path

python 5a12_PUL_syn_train_test_split.py

python 5a12_PUL_syn_truncate.py

python 5a12_PUL_exp_train_test_split.py

python 5a12_PUL_exp_truncate.py

python 5a12_2ag_train_test_split.py

python 5a12_2ag_truncate.py