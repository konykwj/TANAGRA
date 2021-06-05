#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 21:31:48 2019

@author: bill

This contains all the functions needed to execute the main NMF Analysis strategy as contained in the NMF_Analysis class.
"""

import pickle
import numpy as np
import scipy.sparse
from sklearn.decomposition import NMF
import sklearn.preprocessing
import scipy

'''
Modifications to H that ensure each topic is mapped to a unit vector in the term space.
'''
def norm_fun(vector):
    return np.linalg.norm(vector) #Normalizing the vector to have a length of one in topic space.

def b_mat(H):
    num_topics = np.shape(H)[0]
    B = np.zeros((num_topics,num_topics), dtype = float)
    B_inv = np.zeros((num_topics,num_topics), dtype = float)
    for topic in range(num_topics):
        norm = norm_fun(H[topic])
        B[topic,topic] = 1/norm
        B_inv[topic,topic] = norm
    return B, B_inv

'''
The main function to run NMF on the desired number of topics. 
'''
def run_ensemble_NMF_strategy(num_topics, num_folds, num_runs, num_docs, doc_term_matrix):

    #Defines the number of elements in each fold and ensures that the total sums correctly
    fold_sizes = (num_docs // num_folds) * np.ones(num_folds, dtype=np.int)
    fold_sizes[:num_docs % num_folds] += 1
    
    #Creates a list that will save all the final H matrices for the last NMF application.
    H_list = []
    
    #For every run over all folds
    for run in range(num_runs):
        doc_ids = np.arange(num_docs)
        np.random.shuffle(doc_ids)
        
        current_fold = 0 
        for fold, fold_size in enumerate(fold_sizes):
            #Updates the currentfold in the process
            start, stop = current_fold, current_fold+fold_size
            current_fold = stop
            
            #Removes the current fold
            sample_ids = list(doc_ids)
            for id in doc_ids[start:stop]:
                sample_ids.remove(id)
            
            #
            sample_doc_ids = []
            for doc_index in sample_ids:
                sample_doc_ids.append(doc_ids[doc_index])
             
            S = doc_term_matrix[sample_ids,:]
            S = scipy.sparse.csr_matrix(S)
            
            model = NMF( init="nndsvd", n_components = num_topics ) 
            W = model.fit_transform( doc_term_matrix )
            H = model.components_                   
            H_list.append(H)
            
            H = 0.0
            W = 0.0
            model = 0.0
            
    M = np.vstack(H_list)
    
    model = NMF( init="nndsvd", n_components = num_topics )
    W = model.fit_transform(M)
    ensemble_H = model.components_ 
    
    HT = sklearn.preprocessing.normalize( ensemble_H.T, "l2", axis=0 )
    
    ensemble_W = doc_term_matrix.dot(HT)
    
    #Updating the W and H matrices to normalize H.
    B,B_inv = b_mat(ensemble_H)
    ensemble_H = np.matmul(B,ensemble_H)
    ensemble_W = np.matmul(ensemble_W, B_inv)
        
    print(num_topics, 'th topic analyzed')
    
    return num_topics, ensemble_W, ensemble_H