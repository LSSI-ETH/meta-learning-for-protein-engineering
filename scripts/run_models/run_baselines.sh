#!/bin/bash

cd ../../baselines/traditional_baselines

python 4d5_syn_traditional_baselines.py

python 4d5_exp_traditional_baselines.py

python 5a12_PUL_syn_traditional_baselines.py

python 5a12_PUL_exp_traditional_baselines.py

python 5a12_2ag_traditional_baselines.py

cd ../elkanoto

python elkanoto_syn.py

python elkanoto_exp.py

cd ../pudms

R CMD BATCH install_pudms_packages.R

python pudms_preprocessing_syn.py

python pudms_preprocessing_exp.py

R CMD BATCH PUDMS_syn.R

R CMD BATCH PUDMS_exp.R

python pudms_process_output_syn.py

python pudms_process_output_exp.py