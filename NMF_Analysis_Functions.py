#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 21:31:48 2019

@author: Bill Konyk

This contains all the functions needed to execute the main NMF Analysis strategy as contained in the NMF_Analysis class.

The process follows the method described in https://arxiv.org/pdf/1702.07186.pdf
"""

import numpy as np
import scipy.sparse
from sklearn.decomposition import NMF
import sklearn.preprocessing
import scipy

'''
Modifications to H that ensure each topic is mapped to a unit vector in the term space.
'''
def norm_fun(vector):
    """
    Calculates the norm of a vector
    
    Parameters
    ----------
    vector : np array
        Some vector
        
    Returns
    -------
    norm : float
        Norm of the vector
    """
    
    return np.linalg.norm(vector)

def b_mat(H):
    """
    Defines the B matrix so that H is normalized to unit length. THis exploits the fact that H B B_inv W = H W
    Note that B is diagonal, so the inverse is simple to define and calculate
    
    Parameters
    ----------
    H : np array
        H matrix from the NMF process
        
    Returns
    -------
    B : np array
        B matrix
    B_inv : np array
        Inverse B matrix
    """
    
    num_topics = np.shape(H)[0]
    B = np.zeros((num_topics,num_topics), dtype = float) #Create matrices
    B_inv = np.zeros((num_topics,num_topics), dtype = float) #Create inverse matrix
    
    for topic in range(num_topics):
        norm = norm_fun(H[topic])
        B[topic,topic] = 1/norm
        B_inv[topic,topic] = norm
        
    return B, B_inv


def run_ensemble_NMF_strategy(num_topics, num_folds, num_runs, doc_term_matrix):
    
    """
    Main function to process text using NMF.
    This implements the method described in https://arxiv.org/pdf/1702.07186.pdf
    It also normalizes the H matrix so that each topic has a norm of length 1
        
    
    Parameters
    ----------
    num_topics : int
        Number of topics to generate
    num_folds : int
        Number of times to partition the set of documents. In each run one of the folds will randomly be excluded
    num_runs : int
        Number of times to run NMF
    doc_term_matrix : np.array
        Vectorized document-term matrix from preprocessing
        
    Returns
    -------
    ensemble_W : sparse matrix
        Sparse form of the W matrix
    ensemble_H : sparse matrix
        Sparse form of the H matrix
    """
    
    #Identify number of documents
    num_docs = doc_term_matrix.shape[0]

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
    
    return ensemble_W, ensemble_H