# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 01:49:36 2021

@author: konyk
"""

import pickle
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize


matplotlib.use('Agg')

from NMF_Analysis import NMF_Analysis


def calc_connectitivity(data_df, doc_topic_mats, data_column, valid_values):
    
    df_list = []

    for doc_topic_mat in doc_topic_mats:
        
        data_list = []
        num_docs, num_topics = doc_topic_mat.shape
        
        for category in valid_values:
            
            valid_docs = data_df['doc_id'][data_df[data_column] == category].unique()
            
            category_dt_mat = doc_topic_mat[valid_docs]
            
                        
            for doc_id in valid_docs:
            
                assigned_topics = np.where(doc_topic_mat[doc_id]>0)[0]
                                
                #Normalizes it so that every document has a sum of one
                count_mat = normalize(category_dt_mat[:, assigned_topics], norm = 'l1', axis = 1)
                
                value = np.sum(count_mat) - len(assigned_topics)
                value = max(value, 0) #In the event there are no shared documents
                
                percent = value/len(valid_docs)*100
                
                data_list.append([doc_id, category, value, num_topics, percent, len(valid_docs)])
                        
        df_list.append(pd.DataFrame(data_list, columns = ['doc_id', data_column, 'value', 'num_topics', 'percent', 'num_docs_in_cat']))
        
    #Adding up all into one
    plot_df = pd.concat(df_list)
    
    #Calculating statistics on the data
    stat_df = pd.DataFrame(plot_df.groupby(['num_topics','category']).agg({'value':['min','max','mean','median'],
                                                                           'percent':['min','max','mean','median']})).reset_index()
    
    return plot_df, stat_df


#%%
'''
Setting up the information needed to run the project
'''
k_topics_list = [i for i in range(3,25)]

num_runs = 10
num_folds = 10
run_name = 'BBC_Greene'

#Create the class to handle the data
nmf = NMF_Analysis('.', run_name, k_topics_list)

data_df = nmf.load_corpus_df()

data_column = 'category'

valid_categories = data_df['category'].unique()




#%% 

'''
Calculating Entropy... lower is better, less suprise

Calculate for each valid topic
'''


dt_list = []

for num_topics in k_topics_list:
    
    filename = nmf.doc_topic_filename(num_topics)
    
    (doc_topic_mat, H, entropy, invalid_docs) = pickle.load(open(filename,'rb'))
    
    dt_list.append(doc_topic_mat)

plot_df, stat_df = calc_connectitivity(data_df, dt_list, 'category', valid_categories)



#%%
'''
Plotting using the top value from the W matrix only...
'''


dt_list = []

for num_topics in k_topics_list:
    
    filename = nmf.w_filename(num_topics)
    
    (doc_topic_mat) = pickle.load(open(filename,'rb'))
    
    max_indices = np.argmax(doc_topic_mat, axis = 1)
    
    for row, column in enumerate(max_indices):
        
        doc_topic_mat[row] = 0
        
        doc_topic_mat[row, column] = 1
    
    dt_list.append(doc_topic_mat)
    
w_plot_df, w_stat_df = calc_connectitivity(data_df, dt_list, 'category', valid_categories)


#%%
'''
Plotting purities
'''

plot_save_path = './'
filetype = '.pdf'


fig, (ax1, ax2) = plt.subplots(1,2, figsize=(17, 7))

# ax1.scatter(plot_df['num_topics'].values, plot_df['value'].values, color = '#CC4F1B', alpha = .3)

# ax1.fill_between(stat_df['num_topics'].values, stat_df['value']['min'].values, stat_df['value']['max'].values,
#     alpha=0.1, edgecolor='#CC4F1B', facecolor='#FF9848')

# color_list = tab20

cmap = plt.get_cmap("tab10")
color_list = [cmap(i) for i in range(len(valid_categories))]


#Looking at the differences between the two methods;


for category, color in zip(stat_df['category'].unique(), color_list):
    
    df = stat_df[stat_df['category'] == category]
    
    ax1.plot(df['num_topics'].values, df['percent']['median'].values, label = category, color = color)
    
    
for category, color in zip(stat_df['category'].unique(), color_list):
    
    df = stat_df[stat_df['category'] == category]
    w_df = w_stat_df[w_stat_df['category'] == category]
    
    data = df['percent']['median'].values - w_df['percent']['median'].values 
    
    ax2.plot(df['num_topics'].values, data, label = category, color = color)
    
    

ax1.set_ylabel('Median Percent of Docs Clustered')
ax1.legend()
ax1.set_xlabel('Topic Number')

ax2.set_ylabel('Difference between median doc clustering')
# ax2.legend()
ax2.set_xlabel('Topic Number')

ax1.grid(which='major', axis='both', linestyle='--')

fig.tight_layout()
fig.savefig(plot_save_path+run_name+'_num_docs_correct' +filetype, dpi = 300)


plt.clf()
plt.close('all')

#%%

'''
Calculating the percent of documents that have been successfully clustered... treating the number of topics differently...

A document is clustered if it is connected to more than 50% of other documents? Or some other metric like a threshold of 20 documents or something?
'''

#%%

#Plotting

#Defining success
w_plot_df['success'] = w_plot_df.apply(lambda x: 1 if x['percent']> 50 else 0, axis = 1)
plot_df['success'] = plot_df.apply(lambda x: 1 if x['percent']> 50 else 0, axis = 1)


plot_save_path = './'
filetype = '.pdf'

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(17, 7))

cmap = plt.get_cmap("tab10")
color_list = [cmap(i) for i in range(len(valid_categories))]

#Plotting overall success
val_df = pd.DataFrame(plot_df.groupby(['num_topics']).agg({'doc_id':['count'],
                                                                      'success':['sum']})).reset_index()
val_df['percent'] = val_df['success']['sum']/val_df['doc_id']['count']*100

w_val_df = pd.DataFrame(w_plot_df.groupby(['num_topics']).agg({'doc_id':['count'],
                                                                      'success':['sum']})).reset_index()
w_val_df['w_percent'] = w_val_df['success']['sum']/w_val_df['doc_id']['count']*100

w_val_df = w_val_df.merge(val_df, on = 'num_topics')

ax1.plot(val_df['num_topics'], val_df['percent'], color = 'k', linestyle = ':', zorder = 3, label = 'Overall Success', linewidth = 3)
ax2.plot(w_val_df['num_topics'], w_val_df['percent'] - w_val_df['w_percent'], color = 'k', linestyle = ':', zorder = 3, label = 'Overall Success', linewidth = 3)


#Plotting success by category
val_df = pd.DataFrame(plot_df.groupby(['num_topics', 'category']).agg({'doc_id':['count'],
                                                                      'success':['sum']})).reset_index()
val_df['percent'] = val_df['success']['sum']/val_df['doc_id']['count']*100

w_val_df = pd.DataFrame(w_plot_df.groupby(['num_topics', 'category']).agg({'doc_id':['count'],
                                                                      'success':['sum']})).reset_index()

w_val_df['w_percent'] = w_val_df['success']['sum']/w_val_df['doc_id']['count']*100

w_val_df = w_val_df.merge(val_df, on = ['num_topics', 'category'])
w_val_df['diff'] = w_val_df['percent'] - w_val_df['w_percent']


for category, color in zip(val_df['category'].unique(), color_list):
    
    df = val_df[val_df['category'] == category]
    w_df = w_val_df[w_val_df['category'] == category]
    
    ax1.plot(df['num_topics'].values, df['percent'].values, label = category, color = color, zorder = -1, alpha = .7)
    ax2.plot(w_df['num_topics'].values, w_df['diff'].values, label = category, color = color, zorder = -1, alpha = .7)


ax1.set_ylabel('Success rate')
ax1.legend()
ax1.set_xlabel('Topic Number')
ax1.set_ylim(0,100)

ax2.set_ylabel('Difference in successes')
ax2.set_xlabel('Topic Number')
ax2.set_ylim(0,100)


ax1.grid(which='major', axis='both', linestyle='--')

fig.tight_layout()
fig.savefig(plot_save_path+run_name+'_success_rate' +filetype, dpi = 300)


plt.clf()
plt.close('all')
