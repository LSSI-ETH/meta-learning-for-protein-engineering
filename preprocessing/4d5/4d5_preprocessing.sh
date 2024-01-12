#!/bin/bash

cd $preprocessing_path

python 4d5_syn_train_test_split.py

python 4d5_syn_truncate.py

python 4d5_exp_train_test_split.py

python 4d5_exp_truncate.py