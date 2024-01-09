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


path = '../../results/'
edit_distance = 'all'

#======================= 4D5 Synthetic Noise ========================================
data_type = '4d5_syn'
data = pd.read_csv(path + f'{data_type}_traditional_baseline.csv')
data = add_metric_to_df(data, is_float = True, metric_str = 'mcc')
data = data.rename(columns={'top_model':'model', 'base_model':'basemodel', 
                            'meta_set_number': 'meta_set' })

data['model'] = data['model'].replace({'RF': 'Random Forest', 'SVM': 'Linear SVC',
                                                         'NB': 'Naive Bayes' })

data = data[data['edit_distance'] == 6]
#=================== Figure ==============================================
sns.set_theme(style="darkgrid")
sns.set(rc={'figure.figsize':(7,5)})
plt.figure()

results = sns.relplot(data= data, kind = 'line', x="trunc", y="best_mcc",
                      hue='noise_fraction', col = "model", row = "edit_distance", marker = "o", palette='viridis_r', 
                      legend = 'full', 
                      col_order = ['Random Forest', 'Linear SVC', 'Naive Bayes'])

(results.set_axis_labels("Fraction Total Data", "Test MCC")
  .set_titles("ED {row_name}: {col_name}")
  .tight_layout(w_pad=0))

results.set(ylim=(-1.0, 1.0))

results.legend.remove()
lgd = results.fig.legend(handles=results.legend.legendHandles[1:10], frameon= False, 
                   title='Noise Fraction (η)', loc='center left', bbox_to_anchor=(-0.075, 0.5), ncol=1)

plt.subplots_adjust(wspace=0.1)
fig_name_str = f'{data_type}_traditional_baselines_ed_{edit_distance}'
plt.savefig(f'{fig_name_str}.png', dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.savefig(f'{fig_name_str}.svg', bbox_extra_artists=(lgd,), bbox_inches='tight')



#======================= 4D5 Experimental Noise ========================================
data_type = '4d5_exp'
data = pd.read_csv(path + f'{data_type}_traditional_baseline.csv')
data = add_metric_to_df(data, is_float = True, metric_str = 'mcc')
data = data.rename(columns={'top_model':'model', 'base_model':'basemodel', 
                            'meta_set_number': 'meta_set' })

data['model'] = data['model'].replace({'RF': 'Random Forest', 'SVM': 'Linear SVC',
                                                         'NB': 'Naive Bayes' })

data['meta_set'] = data['meta_set'].replace({0: 32, 1: 96, 2: 288, 3: 864})
#=================== Figure ==============================================
sns.set_theme(style="darkgrid")
sns.set(rc={'figure.figsize':(7,5)})
plt.figure()

results = sns.relplot(data= data, kind = 'line', x="trunc", y="best_mcc",
                      hue='meta_set', col = "model", row = "edit_distance", marker = "o", palette='nipy_spectral', 
                      legend = 'full', 
                      col_order = ['Random Forest', 'Linear SVC', 'Naive Bayes'])

(results.set_axis_labels("Fraction Total Data", "Test MCC")
  .set_titles("ED {row_name}: {col_name}")
  .tight_layout(w_pad=0))

results.set(ylim=(0.0, 1.0))

results.legend.remove()
lgd = results.fig.legend(handles=results.legend.legendHandles[1:9], frameon= False, 
                   title='Meta Set Size', loc='center left', bbox_to_anchor=(-0.075, 0.5), ncol=1)

plt.subplots_adjust(wspace=0.1)
fig_name_str = f'{data_type}_traditional_baselines_ed_{edit_distance}'
plt.savefig(f'{fig_name_str}.png', dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.savefig(f'{fig_name_str}.svg', bbox_extra_artists=(lgd,), bbox_inches='tight')



#======================= 5a12 Synthetic PUL ========================================
data_type = '5a12_PUL_syn'
data = pd.read_csv(path + f'{data_type}_traditional_baseline.csv')
data = add_metric_to_df(data, is_float = True, metric_str = 'mcc')
data = data.rename(columns={'top_model':'model', 'base_model':'basemodel', 
                            'meta_set_number': 'meta_set' })

data['model'] = data['model'].replace({'RF': 'Random Forest', 'SVM': 'Linear SVC',
                                                         'NB': 'Naive Bayes' })
data = data[data['edit_distance'] == 5]
#=================== Figure ==============================================
sns.set_theme(style="darkgrid")
sns.set(rc={'figure.figsize':(7,5)})
plt.figure()

results = sns.relplot(data= data, kind = 'line', x="trunc", y="best_mcc",
                      hue='alpha', col = "model", row = "edit_distance", marker = "o", palette='nipy_spectral', 
                      legend = 'full',
                      col_order = ['Random Forest', 'Linear SVC', 'Naive Bayes'])

(results.set_axis_labels("Fraction Total Data", "Test MCC")
  .set_titles("ED {row_name}: {col_name}")
  .tight_layout(w_pad=0))

results.set(ylim=(0.0, 1.0))

results.legend.remove()
lgd = results.fig.legend(handles=results.legend.legendHandles[1:10], frameon= False, 
                   title='% P in U (α)', loc='center left', bbox_to_anchor=(-0.075, 0.5), ncol=1)

plt.subplots_adjust(wspace=0.1)
fig_name_str = f'{data_type}_traditional_baselines_ed_{edit_distance}'
plt.savefig(f'{fig_name_str}.png', dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.savefig(f'{fig_name_str}.svg', bbox_extra_artists=(lgd,), bbox_inches='tight')


#======================= 5a12 Experimental PUL ========================================
data_type = '5a12_PUL_exp'
data = pd.read_csv(path + f'{data_type}_traditional_baseline.csv')
data = add_metric_to_df(data, is_float = True, metric_str = 'mcc')
data = data.rename(columns={'top_model':'model', 'base_model':'basemodel', 
                            'meta_set_number': 'meta_set' })

data['model'] = data['model'].replace({'RF': 'Random Forest', 'SVM': 'Linear SVC',
                                                         'NB': 'Naive Bayes' })

data['meta_set'] = data['meta_set'].replace({0: 32, 1: 96, 2: 288, 3: 864})
#=================== Figure ==============================================
sns.set_theme(style="darkgrid")
sns.set(rc={'figure.figsize':(7,5)})
plt.figure()

results = sns.relplot(data= data, kind = 'line', x="trunc", y="best_mcc",hue='meta_set', 
                      col = "model", row = "edit_distance", marker = "o", palette='Spectral', legend = 'full',
                      col_order = ['Random Forest', 'Linear SVC', 'Naive Bayes'])
                      
(results.set_axis_labels("Fraction Total Data", "Test MCC")
  .set_titles("ED {row_name}: {col_name}")
  .tight_layout(w_pad=0))

results.set(ylim=(0.0, 1.0))

results.legend.remove()
lgd = results.fig.legend(handles=results.legend.legendHandles[:10], frameon= False, 
                   title='Meta Set Size', loc='center left', bbox_to_anchor=(-0.075, 0.5), ncol=1)

plt.subplots_adjust(wspace=0.1)
fig_name_str = f'{data_type}_traditional_baselines_ed_{edit_distance}'
plt.savefig(f'{fig_name_str}.png', dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.savefig(f'{fig_name_str}.svg', bbox_extra_artists=(lgd,), bbox_inches='tight')


#======================= 5a12 2ag ================================================
data_type = '5a12_2ag'
data = pd.read_csv(path + f'{data_type}_traditional_baseline.csv')
data = add_metric_to_df(data, is_float = True, metric_str = 'mcc')
data = data.rename(columns={'top_model':'model', 'base_model':'basemodel', 
                            'meta_set_number': 'meta_set' })

data['model'] = data['model'].replace({'RF': 'Random Forest', 'SVM': 'Linear SVC',
                                                         'NB': 'Naive Bayes' })

data['meta_set'] = data['meta_set'].replace({0: 32, 1: 96, 2: 288, 3: 864})
#=================== Figure ==============================================

sns.set_theme(style="darkgrid")
sns.set(rc={'figure.figsize':(7,5)})
plt.figure()

results = sns.relplot(data= data, kind = 'line', x="trunc", y="best_mcc",hue='meta_set', 
                      col = "model", row = "edit_distance", marker = "o", palette='plasma_r', legend = 'full',
                      col_order = ['Random Forest', 'Linear SVC', 'Naive Bayes'])
                      

(results.set_axis_labels("Fraction Total Data", "Test MCC")
  .set_titles("ED {row_name}: {col_name}")
  .tight_layout(w_pad=0))

results.set(ylim=(0.0, 1.0))


results.legend.remove()
lgd = results.fig.legend(handles=results.legend.legendHandles[:10], frameon= False, 
                   title='Meta Set Size', loc='center left', bbox_to_anchor=(-0.075, 0.5), ncol=1)

plt.subplots_adjust(wspace=0.1)
fig_name_str = f'{data_type}_traditional_baselines_ed_{edit_distance}'
plt.savefig(f'{fig_name_str}.png', dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.savefig(f'{fig_name_str}.svg', bbox_extra_artists=(lgd,), bbox_inches='tight')
