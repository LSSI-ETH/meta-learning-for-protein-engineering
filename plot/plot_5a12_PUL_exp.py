#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns

def add_metric_to_df(df, is_float, metric_str):
    #if run was executed with CUDA, parse metric from string
    data = df.copy()
    modified_metric = []
    
    if not is_float:
        for entry in data['output/test_' + metric_str]:
            m = re.search(r'[\-]*0\.[0-9]{4}',entry)
            try:
                modified_metric.append(float(m.group(0)))
            except:
                if entry.startswith('nan'):
                    modified_metric.append(np.nan)
                else:
                    modified_metric.append(np.nan)
        data['best_' + metric_str] = modified_metric
    else:
        data['best_' + metric_str] = data['output/test_' + metric_str]
    return data

data_type = '5a12_PUL_exp'
path = '../results/'
transformer = f'{data_type}_transformer_2.csv'
cnn = f'{data_type}_cnn.csv'
logistic_regression = f'{data_type}_logistic_regression.csv'
mlp = f'{data_type}_mlp.csv'

pudms = f'{data_type}_pudms.csv'
elk_rf = f'{data_type}_Elkanoto_RF.csv'
elk_lr = f'{data_type}_Elkanoto_LR.csv'

model_list = [cnn, transformer, logistic_regression, mlp, pudms, elk_rf, elk_lr]
model_str_list = ['cnn', 'transformer', 'logistic_regression', 'mlp', 'pudms', 'elk_rf', 'elk_lr']

data = pd.DataFrame()
for model, model_str in zip(model_list, model_str_list):

    tmp_data = pd.read_csv(path + model)    
    
    if model_str == 'cnn' or model_str == 'transformer': isfloat = False
    elif model_str == 'logistic_regression' or model_str == 'mlp': isfloat = True
    else: isfloat = True
    
    tmp_data = add_metric_to_df(tmp_data, is_float = isfloat, metric_str = 'mcc')
    data = pd.concat([data, tmp_data], ignore_index = True)


data = data.rename(columns={'top_model':'model', 'base_model':'basemodel', 
                            'meta_set_number': 'meta_set' })


data['meta_set'] = data['meta_set'].replace({0: 32, 1: 96, 2: 288, 3: 864})


baselines_pul = data[(data['basemodel'] == 'pudms') | 
                 (data['basemodel'] == 'elkanoto') ]

data = pd.merge(data,baselines_pul, how='outer', indicator=True)
data = data[data._merge.ne('both')].drop('_merge',1)



data['model'] = data['model'].replace({'fine-tune': 'FT Baseline',
                                       'metaset_baseline': 'MSO Baseline'})

data['model'] = data['model'].replace({'l2rw': 'L2RW', 'mlc': 'MLC',
                                       'standard': 'Standard'})

data['basemodel'] = data['basemodel'].replace({'transformer_2': 'Transformer', 
                                               'cnn': 'CNN', 
                                               'logistic_regression': 'Logistic Regression',
                                               'mlp': 'MLP'})


baselines_pul['model'] = baselines_pul['model'].replace({'Elkanoto_RF': 'ElkaNoto RF', 
                                                         'Elkanoto_LR': 'ElkaNoto LR',
                                                         'pudms_order_1': 'PUDMS Order 1',
                                                         'pudms_order_2': 'PUDMS Order 2',})



#=================== PU Learning Baselines ========================================
sns.set_theme(style="darkgrid")
sns.set(rc={'figure.figsize':(7,5)})
plt.figure() 

results = sns.relplot(data= baselines_pul, kind = 'line', 
                      x="trunc", y="best_mcc", hue='meta_set',
                      col = "model", marker = "o", legend='full', 
                      row = 'edit_distance',
                      palette = 'Spectral', 
                      col_order = ['PUDMS Order 1', 'PUDMS Order 2', 
                                   'ElkaNoto RF', 'ElkaNoto LR'])
                      
(results.set_axis_labels("Fraction Total Data", "Test MCC")
  .set_titles("{col_name}")
  .tight_layout(w_pad=0))

results.set(ylim=(0.0, 1.0))

results.legend.remove()
lgd = results.fig.legend(handles=results.legend.legendHandles[:10], frameon= False, 
                   title='Meta Set Size', loc='center left', bbox_to_anchor=(-0.075, 0.5), ncol=1)

plt.subplots_adjust(wspace=0.1)
fig_name_str = f'{data_type}_PUL_baselines_edit_distance'
plt.savefig(f'{fig_name_str}.png', dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.savefig(f'{fig_name_str}.svg', bbox_extra_artists=(lgd,), bbox_inches='tight')





for basemodel in ['CNN', 'Transformer']:
    data_main = data[data['basemodel'] == basemodel]
    sns.set(rc={'figure.figsize':(7,5)})
    plt.figure()
    
    results = sns.relplot(data= data_main, kind = 'line', x="truncate_factor", y="best_mcc",hue='meta_set', 
                          col = "model", row = "edit_distance", marker = "o", palette = 'Spectral', 
                          legend = 'full',
                          col_order = ['Standard', 'FT Baseline', 'L2RW', 'MLC']) 
                          
    

    (results.set_axis_labels("Fraction Total Data", "Test MCC")
      .tight_layout(w_pad=0))
    
    if basemodel == 'CNN':
        results.set_titles("ED {row_name}, CNN: {col_name}")
    elif basemodel == 'Transformer':
        results.set_titles("ED {row_name}, Transformer: {col_name}")
    
    results.set(ylim=(0.3, 1.0))
    
    
    results.legend.remove()
    lgd = results.fig.legend(handles=results.legend.legendHandles[:10], frameon= False, 
                       title='Meta Set Size', loc='center left', bbox_to_anchor=(-0.075, 0.5), ncol=1)
    
    plt.subplots_adjust(wspace=0.1)
    fig_name_str = f'{data_type}_{basemodel}_edit_distance_std'
    plt.savefig(f'{fig_name_str}.png', dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(f'{fig_name_str}.svg', bbox_extra_artists=(lgd,), bbox_inches='tight')
    
    

for basemodel in ['Logistic Regression', 'MLP']:
    data_supp = data[data['basemodel'] == basemodel]
    sns.set(rc={'figure.figsize':(7,5)})
    plt.figure()
    
    results = sns.relplot(data= data_supp, kind = 'line', x="truncate_factor", y="best_mcc",hue='meta_set', 
                          col = "model", row = "edit_distance", marker = "o", palette = 'Spectral', 
                          legend = 'full',
                          col_order = ['Standard', 'FT Baseline', 'L2RW', 'MLC']) 
                          
    

    (results.set_axis_labels("Fraction Total Data", "Test MCC")
      .tight_layout(w_pad=0))
    
    if basemodel == 'Logistic Regression':
        results.set_titles("ED {row_name}, Logistic Regression: {col_name}")
    elif basemodel == 'MLP':
        results.set_titles("ED {row_name}. MLP: {col_name}")
    
    results.set(ylim=(0.0, 1.0))
    
    
    results.legend.remove()
    lgd = results.fig.legend(handles=results.legend.legendHandles[:10], frameon= False, 
                       title='Meta Set Size', loc='center left', bbox_to_anchor=(-0.075, 0.5), ncol=1)
    
    plt.subplots_adjust(wspace=0.1)
    fig_name_str = f'{data_type}_{basemodel}_edit_distance_supp_std'
    plt.savefig(f'{fig_name_str}.png', dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(f'{fig_name_str}.svg', bbox_extra_artists=(lgd,), bbox_inches='tight')

