#!/bin/bash

python 5a12_PUL_syn_train_test_split.py

python 5a12_PUL_syn_truncate.py

python 5a12_PUL_exp_train_test_split.py

python 5a12_PUL_exp_truncate.py

python 5a12_2ag_train_test_split.py

python 5a12_2ag_truncate.py