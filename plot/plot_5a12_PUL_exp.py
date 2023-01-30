#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns

def add_metric_to_df(df, is_float, metric_str):
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
transformer = f'{data_type}_transformer.csv'
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
    elif model_str == 'logistic_regression' or model_str == 'mlp': isfloat = False
    else: isfloat = True
    
    tmp_data = add_metric_to_df(tmp_data, is_float = isfloat, metric_str = 'mcc')
    data = pd.concat([data, tmp_data], ignore_index = True)

baselines_pul = data[(data['basemodel'] == 'pudms') | 
                 (data['basemodel'] == 'elkanoto') ]

data = pd.merge(data,baselines_pul, how='outer', indicator=True)
data = data[data._merge.ne('both')].drop('_merge',1)

data['model'] = data['model'].replace({'fine-tune': 'FT Baseline',
                                                         'metaset_baseline': 'MSO Baseline' })

data['model'] = data['model'].replace({'l2rw': 'L2RW', 'mlc': 'MLC'})
data['basemodel'] = data['basemodel'].replace({'transformer': 'Transformer', 'cnn': 'CNN', 
                                               'logistic_regression': 'Logistic Regression', 'mlp': 'MLP'})


#=================== Main Figure ==============================================
data_main = data[(data['basemodel'] == 'Transformer') | (data['basemodel'] == 'CNN')]
sns.set_theme(style="darkgrid")
sns.set(rc={'figure.figsize':(7,5)})
plt.figure()

results = sns.relplot(data= data_main, kind = 'line', x="truncate_factor", y="best_mcc",hue='meta_len', 
                      col = "model", row = "basemodel", marker = "o", palette='Spectral', legend = 'full',
                      col_order = ['FT Baseline', 'L2RW', 'MLC'])
                      
(results.set_axis_labels("Fraction Total Data", "Test MCC")
  .set_titles("{row_name}: {col_name}")
  .tight_layout(w_pad=0))

results.set(ylim=(0.0, 1.0))

results.legend.remove()
lgd = results.fig.legend(handles=results.legend.legendHandles[:10], frameon= False, 
                   title='Meta Set Size', loc='center left', bbox_to_anchor=(-0.075, 0.5), ncol=1)

plt.subplots_adjust(wspace=0.1)
fig_name_str = f'{data_type}'
plt.savefig(f'{fig_name_str}.png', dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.savefig(f'{fig_name_str}.svg', bbox_extra_artists=(lgd,), bbox_inches='tight')



#=================== PU Learning Baselines ========================================
sns.set_theme(style="darkgrid")
sns.set(rc={'figure.figsize':(7,5)})
plt.figure()

baselines_pul['model'] = baselines_pul['model'].replace({'Elkanoto_RF': 'ElkaNoto RF', 
                                                         'Elkanoto_LR': 'ElkaNoto LR',
                                                         'pudms_order_1': 'PUDMS Order 1',
                                                         'pudms_order_2': 'PUDMS Order 2',})

results = sns.relplot(data= baselines_pul, kind = 'line', x="truncate_factor", y="best_mcc", hue='meta_len',
                      col = "model", marker = "o", legend='full', palette = 'Spectral',
                      col_order = ['PUDMS Order 1', 'PUDMS Order 2', 'ElkaNoto RF', 'ElkaNoto LR'])
                      
(results.set_axis_labels("Fraction Total Data", "Test MCC")
  .set_titles("{col_name}")
  .tight_layout(w_pad=0))

results.set(ylim=(0.0, 1.0))

results.legend.remove()
lgd = results.fig.legend(handles=results.legend.legendHandles[:10], frameon= False, 
                   title='Meta Set Size', loc='center left', bbox_to_anchor=(-0.075, 0.5), ncol=1)

plt.subplots_adjust(wspace=0.1)
fig_name_str = f'{data_type}_PUL_baselines'
plt.savefig(f'{fig_name_str}.png', dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.savefig(f'{fig_name_str}.svg', bbox_extra_artists=(lgd,), bbox_inches='tight')


#=================== Supp Figure ==============================================sns.set_theme(style="darkgrid")
data_supp = data[(data['basemodel'] == 'Logistic Regression') | (data['basemodel'] == 'MLP')]
sns.set(rc={'figure.figsize':(7,5)})
plt.figure()

results = sns.relplot(data= data_supp, kind = 'line', x="truncate_factor", y="best_mcc",hue='meta_len', 
                      col = "model", row = "basemodel", marker = "o", palette='Spectral', legend = 'full',
                      col_order = ['FT Baseline', 'L2RW', 'MLC'])
                      
(results.set_axis_labels("Fraction Total Data", "Test MCC")
  .set_titles("{row_name}: {col_name}")
  .tight_layout(w_pad=0))

results.set(ylim=(0.0, 1.0))

results.legend.remove()
lgd = results.fig.legend(handles=results.legend.legendHandles[:10], frameon= False, 
                   title='Meta Set Size', loc='center left', bbox_to_anchor=(-0.075, 0.5), ncol=1)

plt.subplots_adjust(wspace=0.1)
fig_name_str = f'{data_type}_supp'
plt.savefig(f'{fig_name_str}.png', dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.savefig(f'{fig_name_str}.svg', bbox_extra_artists=(lgd,), bbox_inches='tight')
