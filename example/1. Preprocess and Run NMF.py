#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 11:12:43 2019

@author: bill

A series of runs splitting everything up in time.
"""


from tanagra import NMF_Analysis
import pickle

'''
Setting up the information needed to run the project
'''
#Variables used for all runs
raw_data_file = './bbc_text_df.pkl'

doc_df = pickle.load(open(raw_data_file,'rb'))

k_topics_list = [i for i in range(3,11)]

num_runs = 10
num_folds = 10
run_name = 'BBC_Greene'

#Create the class to handle the data
nmf = NMF_Analysis('.', run_name, k_topics_list)

# #Vectorize the data and perform preprocessing steps
nmf.vectorize_data(doc_df)

# #Perform the NMF analysis
nmf.run_analysis(num_folds, num_runs)

# #Plot the quality of the data; used to choose the angle threshold
nmf.plot_topic_assignment_scan()


#%%

nmf.assign_topics(40, max_topics = 2)

nmf.plot_assigned_statistics()

#%%
nmf.plot_topic_scan()

#%%
nmf.plot_topic_reports([5,10], 150)
