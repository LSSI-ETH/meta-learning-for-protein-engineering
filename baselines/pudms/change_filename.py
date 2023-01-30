#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 14:28:40 2023

@author: mminot
"""
import pandas as pd
import os
import glob
import re

path = 'predictions_delete/'
outpath = 'predictions/'
extension = 'csv'
result = glob.glob(path + '*.{}'.format(extension))

for file in result:
    print(file)
    tmp_df =pd.read_csv(file)
    m = re.search(r'predictions_delete/(.*)_pyopt.*',file)
    print(m.group(1))#model_list.append(m.group(1))
    tmp_df.to_csv(outpath + m.group(1) + '.csv', index = False, header = True)


