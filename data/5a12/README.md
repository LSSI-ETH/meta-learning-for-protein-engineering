# 5A12 VEGF and ANG-2 Binding Classification Data

This folder contains the 5A12 scFv binding data for two antigens: VEGF and ANG-2. The raw data files with 'sort' in the title are preprocessed from the NGS runs of different FACS screening rounds of the yeast display 5A12 scFv library for binding to the respective antigens. The sequences were experimentally binned into High, Low/Non-Binding to antigen. 

To facilitate ease of use, datasets prepared for machine learning are planned for upload. In the meantime, to reproduce the paper results the preprocessing pipeline can be executed as described on the main page of this repo.

The learning two learning tasks associated with the 5A12 data are Positive and Unlabeled Learning (PUL) and Multi-Antigen Classification from Largely Single-Antigen Data.  

5A12 PUL:
Raw and preprocessed NGS data includes a clean dataset from the final FACS rounds and a noisy dataset consisting of the unsorted library. Note the unsorted library contains a mix both High and Low/Non-Binding sequences.

5A12 Multi-Antigen Classification from Largely Single-Antigen Data:
Raw and preprocessed NGS data includes final FACS rounds of 5A12 screened for binding to either VEGF or ANG-2.
