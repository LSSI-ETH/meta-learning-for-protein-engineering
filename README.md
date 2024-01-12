![Python Version: 3.8.5](https://img.shields.io/badge/Python%20Version-3.8.5-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen)](https://opensource.org/licenses/MIT)

# Meta Learning Improves Robustness and Performance in Machine Learning-Guided Protein Engineering

This repository contains code to perform the analysis described Minot & Reddy 2024. 
[[Cell Systems](https://doi.org/10.1016/j.cels.2023.12.003)] [[bioRxiv](https://www.biorxiv.org/content/10.1101/2023.01.30.526201v1)].

## Table of contents
1. [Prepare Working Environment](#prepare-working-environment)
2. [Datasets](#datasets)
3. [Reproducing Study Results](#reproducing-study-results)
4. [Citing This Work](#citing-this-work)
5. [Citing Supporting Repositories](#citing-supporting-repositories)
6. [License](#license)


## Prepare Working Environment

#### Setup with Conda

```console
conda env create -f meta_env.yml
conda activate meta_env
```
#### Setup with venv
For virtualenv setup:
1. `python -m venv meta_env`
2. Windows:
`meta_env\Scripts\activate.bat`
Or Unix / MacOS:
`source meta_env/bin/activate`
3. `pip install -r requirements.txt`

Note: study results were executed with torch 1.11.0+cu113. Environment contains torch 1.11.0

## Datasets 


The following data is provided in `data/` to facilitate ease of use:
1. preprocessed NGS data in `data/4d5` and `data/5a12`
2. Fully processed train, meta, validation, and test sets are planned for upload. In the meantime, one can curate the data into train, val, meta, and test sets for each learning task via the preprocessing pipeline described in Step 1 below.

## Reproducing Study Results

The full pipeline to reproduce the study, written in Python, can be summarised into three consecutive steps:

 1. Data preprocessing.
 2. Model training and evaluation.
 3. Plot results.

### Step 1 - Preprocessing

To download and start from raw deep sequencing (NGS) data, execute step A). To start from preprocessed NGS data contained in this repo, skip A) and head to step B). In `scripts/`:

#### A) Raw Data

Raw data is available at [[NCBI](https://www.ncbi.nlm.nih.gov/bioproject/?term=1029889)]
or on Mendeley in two parts: [[Part 1](https://data.mendeley.com/datasets/56h8gp94fj/1)] and [[Part 2](https://data.mendeley.com/datasets/s829ccsg6f/1)].
Save data to `data/raw_ngs/`.

After installing [[BBDUK](https://jgi.doe.gov/data-and-tools/software-tools/bbtools/bb-tools-user-guide/installation-guide/)]. Modify the path to BBDUK and this repo before executing `5a12_preprocessing_raw_ngs.sh` and `4d5_preprocessing_raw_ngs.sh` in `preprocessing/`. Then proceed to B).


#### B) From preprocessed NGS:

To execute preprocessing and train/test splitting for each task, in `preprocessing/`, run `5a12_preprocessing.sh` and `4d5_preprocessing.sh`.
Processed datasets for the full analysis in the paper will then be added to `data/`.


### Step 2 - Model Training and Evaluation

Run scripts in `scripts/run_models/`. This will populate the folder `results/` with .csv files in the appropriate format for plotting in Step 3.


### Setp 3 - Plot Results

Run python scripts in `plot/`:

## Citing this Work

Please cite our work when referencing this repository.

```
@article{minot_meta_2024,
	title = {Meta learning addresses noisy and under-labeled data in machine learning-guided antibody engineering},
	issn = {2405-4712},
	url = {https://www.sciencedirect.com/science/article/pii/S2405471223003332},
	doi = {https://doi.org/10.1016/j.cels.2023.12.003},
	journal = {Cell Systems},
	author = {Minot, Mason and Reddy, Sai T.},
	year = {2024},
}
```

## Citing Supporting Repositories
For L2RW [[GitHub](https://github.com/uber-research/learning-to-reweight-examples)]:

```
@inproceedings{ren_learning_2018,
	title = {Learning to {Reweight} {Examples} for {Robust} {Deep} {Learning}},
	url = {https://proceedings.mlr.press/v80/ren18a.html},
	booktitle = {Proceedings of the 35th {International} {Conference} on {Machine} {Learning}},
	publisher = {PMLR},
	author = {Ren, Mengye and Zeng, Wenyuan and Yang, Bin and Urtasun, Raquel},
	month = jul,
	year = {2018},
	note = {ISSN: 2640-3498},
	pages = {4334--4343},
}
```

For MLC [[GitHub](https://github.com/microsoft/MLC)]:

```
@article{zheng_meta_2021,
	title = {Meta {Label} {Correction} for {Noisy} {Label} {Learning}},
	volume = {35},
	copyright = {Copyright (c) 2021 Association for the Advancement of Artificial Intelligence},
	issn = {2374-3468},
	url = {https://ojs.aaai.org/index.php/AAAI/article/view/17319},
	doi = {10.1609/aaai.v35i12.17319},
	journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
	author = {Zheng, Guoqing and Awadallah, Ahmed Hassan and Dumais, Susan},
}
```

For PUDMS [[GitHub](https://github.com/RomeroLab/pudms)]:

```
@article{SONG202192,
title = {Inferring Protein Sequence-Function Relationships with Large-Scale Positive-Unlabeled Learning},
journal = {Cell Systems},
volume = {12},
number = {1},
pages = {92-101.e8},
year = {2021},
issn = {2405-4712},
doi = {https://doi.org/10.1016/j.cels.2020.10.007},
url = {https://www.sciencedirect.com/science/article/pii/S2405471220304142},
author = {Hyebin Song and Bennett J. Bremer and Emily C. Hinds and Garvesh Raskutti and Philip A. Romero},
}
```

For ElkaNoto [[GitHub](https://github.com/pulearn/pulearn)]:

```
@inproceedings{elkan_learning_2008,
	address = {New York, NY, USA},
	series = {{KDD} '08},
	title = {Learning classifiers from only positive and unlabeled data},
	isbn = {978-1-60558-193-4},
	url = {https://doi.org/10.1145/1401890.1401920},
	doi = {10.1145/1401890.1401920},
	booktitle = {Proceedings of the 14th {ACM} {SIGKDD} international conference on {Knowledge} discovery and data mining},
	publisher = {Association for Computing Machinery},
	author = {Elkan, Charles and Noto, Keith},
	month = aug,
	year = {2008},
}
```

## License
The licenses for the software used in this study can be found on the respective githubs as well and in `license/`