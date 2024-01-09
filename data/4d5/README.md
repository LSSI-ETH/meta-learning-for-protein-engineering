# 4D5 Trastuzumab Binding Classification Data

This folder contains the 4D5 scFv trastuzumab binding data. The raw data files with 'sort' in the title are preprocessed from the NGS runs of different FACS screening rounds of the yeast display 4D5 scFv library for binding to trastuzumab. The sequences were experimentally binned into High, Low, and Non-Binding to trastuzumab. 

To facilitate ease of use, datasets prepared for machine learning are planned for upload. In the meantime, to reproduce the paper results the preprocessing pipeline can be executed as described on the main page of this repo.

The learning tasks associated with the 4D5 data include a clean dataset from the final FACS rounds and a noisy dataset utilizing the first FACS round for train and the final FACS round for meta and test sets.